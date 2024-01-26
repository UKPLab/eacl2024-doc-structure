import logging
from typing import Any, ClassVar, List, Dict, Union, Optional, Callable

import omegaconf
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from pytorch_lightning.core.optimizer import LightningOptimizer
import transformers

from evaluation.common import BaseModel, Statistics, BaseInstance
from evaluation.tasks import evidence_inference
from evaluation.tasks.qasper import QASPERTask, QASPERPrediction
from evaluation.tasks.evidence_inference import (
    EvidenceInferenceTask,
    EvidenceInferencePrediction,
    LABEL_TO_CODE
)
from evaluation.tasks.scaffold_task_for_pretraining import ScaffoldTaskForPreTrainingPrediction
from structformer.input_sequence import to_input_sequence, get_structural_tokens
from structformer.position_embeddings import (
    get_post_encoder_position_embedding_ids,
    LongT5StructuralPositionEmbedding
)
from structformer.scaffold_tasks import S2ORCITGSubsetInstance, collate_s2orcitgsubset_instances, ScaffoldTasksHead, make_scaffold_tasks_labels_and_mask
from structformer.tokenized_structure import make_tokenized_structure
from structformer.t5_masking import get_n_noise_spans, t5_mask_input_and_get_output

logger = logging.getLogger(__name__)

class LongT5ForQASPERModel(BaseModel):
    task_name: ClassVar[str] = QASPERTask.task_name
    model_name: ClassVar[str] = "LongT5"

    def __init__(self, config: omegaconf.DictConfig, stats: Statistics):
        super(LongT5ForQASPERModel, self).__init__(config, stats)

        # set up tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/long-t5-tglobal-base"
        )
        self.tokenizer.add_tokens(get_structural_tokens(self.config))

        # set up model
        self.model = transformers.LongT5ForConditionalGeneration.from_pretrained(
            "google/long-t5-tglobal-base",
            use_cache=self.config["use_cache"]
        )

        num_structural_tokens = len(get_structural_tokens(self.config))
        self.model.resize_token_embeddings(self.model.config.vocab_size + num_structural_tokens)

        self.evidence_linear_layer_1 = torch.nn.Linear(
            in_features=self.model.config.d_model,
            out_features=self.model.config.d_model
        )
        self.evidence_linear_layer_2 = torch.nn.Linear(
            in_features=self.model.config.d_model,
            out_features=2  # not evidence (0) and evidence (1)
        )

        # set up loss weights (weights were derived from train data statistics)
        if self.config["model"]["evidence"]["use_evidence_loss_weights"]:
            self.register_buffer("evidence_loss_weight", torch.tensor([1.0197, 51.5619]))
        else:
            self.register_buffer("evidence_loss_weight", torch.tensor([1.0, 1.0]))

        # prepare position embeddings
        self.batch_injector = {}
        if self.config["position_embeddings"]["mode"] != "vanilla":
            logger.info("Using structural position embeddings.")
            # noinspection PyTypeChecker
            self.structural_position_embeddings = LongT5StructuralPositionEmbedding(
                self.config,
                self.model.config.d_model
            )

        # prepare scaffolds
        self.scaffold_tasks_head = ScaffoldTasksHead(
            hidden_dimension=self.model.config.d_model,
            config=self.config
        )

    def training_step(self, batch, batch_idx):
        self.batch_injector["batch"] = batch

        if "is_s2orc_itg_subset_batch" in batch.keys():
            batch["inputs"]["decoder_input_ids"] = torch.ones((len(batch["instances"]), 1), dtype=torch.long, device=self.device)
            output = self.model(**batch["inputs"])

            all_hidden_states = output.encoder_last_hidden_state
            scaffold_loss = self.scaffold_tasks_head(
                all_hidden_states=all_hidden_states,
                scaffold_tasks_mask=batch["scaffold_tasks_mask"],
                scaffold_tasks_node_types_labels=batch["scaffold_tasks_node_types_labels"],
                scaffold_tasks_node_depths_labels=batch["scaffold_tasks_node_depths_labels"]
            )
            return scaffold_loss

        inputs_embeds = self.embed(
            batch
        )

        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch['longt5_inputs']["attention_mask"],
            labels=batch['longt5_inputs']["labels"],
        )

        # evidence detection
        # distinction between instances is lost
        relevant_hidden_states = output.encoder_last_hidden_state[batch["evidence_mask"]]
        evidence_tmp = self.evidence_linear_layer_1(relevant_hidden_states)
        evidence_tmp = torch.tanh(evidence_tmp)
        evidence_logits = self.evidence_linear_layer_2(evidence_tmp)

        loss = output.loss
        if self.config["model"]["evidence"]["learn_evidence_detection"]:
            loss *= (1 - self.config["model"]["evidence"]["evidence_detection_weight"])
            evidence_labels = torch.tensor(
                [label for label_list in batch['evidence_labels'] for label in label_list],
                device=evidence_logits.device
            )
            evidence_loss = F.cross_entropy(
                evidence_logits,
                evidence_labels,
                weight=self.evidence_loss_weight
            )
            evidence_loss *= self.config["model"]["evidence"]["evidence_detection_weight"]
            loss += evidence_loss

        if self.config["scaffold_tasks"]["mode"] != "vanilla" and self.config["scaffold_tasks"]["on_task_data"]:
            loss *= (1 - self.config["scaffold_tasks"]["scaffold_weight"])

            all_hidden_states = output.encoder_last_hidden_state
            scaffold_loss = self.scaffold_tasks_head(
                all_hidden_states=all_hidden_states,
                scaffold_tasks_mask=batch["scaffold_tasks_mask"],
                scaffold_tasks_node_types_labels=batch["scaffold_tasks_node_types_labels"],
                scaffold_tasks_node_depths_labels=batch["scaffold_tasks_node_depths_labels"]
            )
            scaffold_loss *= self.config["scaffold_tasks"]["scaffold_weight"]
            loss += scaffold_loss

        return loss

    def validation_step(self, batch, batch_idx):
        self.batch_injector["batch"] = batch

        del batch["longt5_inputs"]["labels"]

        inputs_embeds = self.embed(batch)

        output = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch['longt5_inputs']['attention_mask'],
            num_beams=self.config["model"]["num_beams"],
            do_sample=self.config["model"]["do_sample"],
            length_penalty=self.config["model"]["length_penalty"],
            max_length=self.config["model"]["max_length"],
            return_dict_in_generate=True,
            output_hidden_states=True
        )

        predictions = []
        for batch_ix, (sequence, instance) in enumerate(zip(output.sequences, batch["instances"])):
            # evidence detection
            relevant_hidden_states = output["encoder_hidden_states"][-1][batch_ix][batch["evidence_mask"][batch_ix]]
            evidence_tmp = self.evidence_linear_layer_1(relevant_hidden_states)
            evidence_tmp = torch.tanh(evidence_tmp)
            evidence_logits = self.evidence_linear_layer_2(evidence_tmp)

            evidence_logits = evidence_logits.detach()
            evidence_codes = torch.argmax(evidence_logits, dim=1)
            evidence_codes = evidence_codes.cpu()

            all_node_ids = set(node.ix for node in instance.document.nodes)
            evidence_nodes = []
            for code, node in zip(evidence_codes, batch["evidence_nodes"][batch_ix]):  # evidence codes/nodes for entire batch
                if int(code) == 1 and node.ix in all_node_ids:  # consider only codes/nodes from this instance
                    evidence_nodes.append(node)

            text = self.tokenizer.decode(
                sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            predictions.append(QASPERPrediction(
                question_id=instance.question_id,
                answer_text=text,
                evidence_nodes=evidence_nodes
            ))
        return predictions

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config["model"]["learning_rate"]
        )

    def embed(
            self,
            batch: Dict[str, Any],
    ) -> torch.Tensor:
        input_ids = batch["longt5_inputs"]["input_ids"]

        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if self.config['position_embeddings']['mode'] == 'node_types':
            structural_position_ids = batch['node_types_labels']
            inputs_embeds += self.structural_position_embeddings(structural_position_ids)
        elif self.config['position_embeddings']['mode'] == 'node_depths':
            structural_position_ids = batch['node_depths_labels']
            inputs_embeds += self.structural_position_embeddings(structural_position_ids)

        return inputs_embeds


    # noinspection PyUnresolvedReferences,PyTypeChecker
    def training_collate_fn(self, instances: List[BaseInstance]) -> Any:
        if isinstance(instances[0], S2ORCITGSubsetInstance):
            return collate_s2orcitgsubset_instances(instances, self.tokenizer, self.config)

        text_and_spans = [to_input_sequence(instance.document, self.config) for instance in instances]
        texts = [text for text, _ in text_and_spans]
        node_spans = [spans for _, spans in text_and_spans]
        questions = [instance.question for instance in instances]
        offsets = [len(instance.question) + 6 for instance in instances]
        answers = [instance.answer_texts[0] for instance in instances]
        input_texts = [f"{question} </s> {text}" for question, text in zip(questions, texts)]

        longt5_inputs = self.tokenizer(
            input_texts,
            text_target=answers,
            return_tensors="pt",
            padding="longest",
            truncation="longest_first",
            max_length=self.config["max_input_length"],
        )

        tokenized_structure = make_tokenized_structure(
            instances=instances,
            node_spans=node_spans,
            inputs=longt5_inputs,
            offsets=offsets,
            config=self.config
        )

        scaffold_tasks_labels_and_mask = make_scaffold_tasks_labels_and_mask(
            tokenized_structure=tokenized_structure,
            config=self.config
        )

        post_encoder_position_embedding_ids = get_post_encoder_position_embedding_ids(
            self.config['post_encoder_position_embeddings']['mode'],
            tokenized_structure
        )

        # create evidence labels and mask
        evidence_labels = []
        evidence_nodes = []
        evidence_mask = torch.zeros_like(longt5_inputs["input_ids"], dtype=torch.bool)

        relevant_node_types = set(self.config["model"]["evidence"]["relevant_node_types"])
        for batch_ix, (span_mapping, instance) in enumerate(zip(node_spans, instances)):
            evidence_node_ids = set(node.ix for node in instance.evidence_nodes[0])
            offset = len(instance.question) + 6
            evidence_labels_for_instance = []
            evidence_nodes_for_instance = []
            for span in span_mapping.spans:
                # put label on first token in node
                token_ix = longt5_inputs.char_to_token(batch_ix, span.left + offset)
                if token_ix is not None:  # it's sometimes none because of truncation
                    if span.content.ntype in relevant_node_types:
                        evidence_mask[batch_ix, token_ix] = True
                        evidence_nodes_for_instance.append(span.content)
                        if span.content.ix in evidence_node_ids:
                            evidence_labels_for_instance.append(1)
                        else:
                            evidence_labels_for_instance.append(0)

            evidence_labels.append(evidence_labels_for_instance)
            evidence_nodes.append(evidence_nodes_for_instance)

        return {
            "instances": instances,
            "node_spans": node_spans,
            "longt5_inputs": longt5_inputs,
            "evidence_mask": evidence_mask,
            "evidence_labels": evidence_labels,
            "evidence_nodes": evidence_nodes,
            "post_encoder_position_embedding_ids": post_encoder_position_embedding_ids,
            **tokenized_structure,
            **scaffold_tasks_labels_and_mask
        }

    def validation_collate_fn(self, instances: List[BaseInstance]) -> Any:
        return self.training_collate_fn(instances)


class LongT5ForEvidenceInferenceModel(BaseModel):
    task_name: ClassVar[str] = EvidenceInferenceTask.task_name
    model_name: ClassVar[str] = "LongT5"

    def __init__(self, config: omegaconf.DictConfig, stats: Statistics):
        super(LongT5ForEvidenceInferenceModel, self).__init__(config, stats)

        # set up tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/long-t5-tglobal-base"
        )
        self.tokenizer.add_tokens(get_structural_tokens(self.config))

        # set up model
        self.model = transformers.LongT5ForConditionalGeneration.from_pretrained(
            "google/long-t5-tglobal-base",
            use_cache=self.config["use_cache"]
        )

        num_structural_tokens = len(get_structural_tokens(self.config))
        self.model.resize_token_embeddings(self.model.config.vocab_size + num_structural_tokens)

        self.evidence_linear_layer_1 = torch.nn.Linear(
            in_features=self.model.config.d_model,
            out_features=self.model.config.d_model
        )
        self.evidence_linear_layer_2 = torch.nn.Linear(
            in_features=self.model.config.d_model,
            out_features=2  # not evidence (0) and evidence (1)
        )

        # set up loss weights (weights were derived from train data statistics)
        # copied from Longformer implementation
        self.register_buffer("classification_loss_weight", torch.tensor([4.1823, 2.1863, 3.2946]))

        if self.config["model"]["evidence"]["use_evidence_loss_weights"]:
            self.register_buffer("evidence_loss_weight", torch.tensor([1.0185, 55.0365]))
        else:
            self.register_buffer("evidence_loss_weight", torch.tensor([1.0, 1.0]))

        self.classification_linear_layer_1 = torch.nn.Linear(
            in_features=self.model.config.d_model,
            out_features=self.model.config.d_model
        )
        self.classification_linear_layer_2 = torch.nn.Linear(
            in_features=self.model.config.d_model,
            out_features=3  # number of labels in evidence inference
        )

        # prepare position embeddings
        self.batch_injector = {}
        if self.config["position_embeddings"]["mode"] != "vanilla":
            logger.info("Using structural position embeddings.")
            # noinspection PyTypeChecker
            self.structural_position_embeddings = LongT5StructuralPositionEmbedding(
                self.config,
                self.model.config.d_model
            )

        # prepare scaffolds
        self.scaffold_tasks_head = ScaffoldTasksHead(
            hidden_dimension=self.model.config.d_model,
            config=self.config
        )

    def forward(self, batch):
        inputs_embeds = self.embed(batch)
        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch['longt5_inputs']["attention_mask"],
            labels=batch['longt5_inputs']["labels"],
            output_hidden_states=True
        )

        # classification
        classification_tmp = self.classification_linear_layer_1(output.decoder_hidden_states[-1][:, 0])
        classification_tmp = torch.tanh(classification_tmp)
        classification_tmp = self.classification_linear_layer_2(classification_tmp)

        # evidence detection
        # Distinction between instances in batch is lost here
        relevant_hidden_states = output.encoder_last_hidden_state[batch["evidence_mask"]]  # distinction between instances in batch is lost
        evidence_tmp = self.evidence_linear_layer_1(relevant_hidden_states)
        evidence_tmp = torch.tanh(evidence_tmp)
        evidence_tmp = self.evidence_linear_layer_2(evidence_tmp)

        return output, classification_tmp, evidence_tmp

    def training_step(self, batch, batch_idx):
        self.batch_injector["batch"] = batch

        if "is_s2orc_itg_subset_batch" in batch.keys():
            output = self.forward(batch)

            all_hidden_states = output.encoder_last_hidden_state
            scaffold_loss = self.scaffold_tasks_head(
                all_hidden_states=all_hidden_states,
                scaffold_tasks_mask=batch["scaffold_tasks_mask"],
                scaffold_tasks_node_types_labels=batch["scaffold_tasks_node_types_labels"],
                scaffold_tasks_node_depths_labels=batch["scaffold_tasks_node_depths_labels"]
            )
            return scaffold_loss

        output, classification_logits, evidence_logits = self.forward(batch)

        loss = F.cross_entropy(
            classification_logits,
            batch["classification_labels"],
            weight=self.classification_loss_weight
        )

        if self.config["model"]["evidence"]["learn_evidence_detection"]:
            loss *= (1 - self.config["model"]["evidence"]["evidence_detection_weight"])
            evidence_labels = torch.tensor(
                [label for label_list in batch['evidence_labels'] for label in label_list],
                device=evidence_logits.device
            )
            evidence_loss = F.cross_entropy(
                evidence_logits,
                evidence_labels,
                weight=self.evidence_loss_weight
            )
            evidence_loss *= self.config["model"]["evidence"]["evidence_detection_weight"]
            loss += evidence_loss

        if self.config["scaffold_tasks"]["mode"] != "vanilla" and self.config["scaffold_tasks"]["on_task_data"]:
            loss *= (1 - self.config["scaffold_tasks"]["scaffold_weight"])

            all_hidden_states = output.encoder_last_hidden_state
            scaffold_loss = self.scaffold_tasks_head(
                all_hidden_states=all_hidden_states,
                scaffold_tasks_mask=batch["scaffold_tasks_mask"],
                scaffold_tasks_node_types_labels=batch["scaffold_tasks_node_types_labels"],
                scaffold_tasks_node_depths_labels=batch["scaffold_tasks_node_depths_labels"]
            )
            scaffold_loss *= self.config["scaffold_tasks"]["scaffold_weight"]
            loss += scaffold_loss

        return loss

    def validation_step(self, batch, batch_idx):
        self.batch_injector["batch"] = batch

        output, classification_logits, evidence_logits = self.forward(batch)

        classification_logits = classification_logits.detach()
        evidence_logits = evidence_logits.detach()

        label_codes = torch.sub(torch.argmax(classification_logits, dim=1), 1)
        evidence_codes = torch.argmax(evidence_logits, dim=1)

        label_codes = label_codes.cpu()
        evidence_codes = evidence_codes.cpu()

        predictions = []
        for batch_ix, (label_code, instance) in enumerate(zip(label_codes, batch["instances"])):
            label_code = int(label_code)
            all_node_ids = set(node.ix for node in instance.document.nodes)
            evidence_nodes = []
            for code, node in zip(evidence_codes,
                                  batch["evidence_nodes"][batch_ix]):  # evidence codes/nodes for entire batch, see above
                if int(code) == 1 and node.ix in all_node_ids:  # consider only codes/nodes from this instance
                    evidence_nodes.append(node)
            predictions.append(EvidenceInferencePrediction(
                pmc_id=instance.pmc_id,
                prompt_id=instance.prompt_id,
                label=evidence_inference.CODE_TO_LABEL[label_code],
                label_code=label_code,
                evidence_nodes=evidence_nodes
            ))
        return predictions

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config["model"]["learning_rate"]
        )

    def embed(
            self,
            batch: Dict[str, Any],
    ) -> torch.Tensor:
        input_ids = batch["longt5_inputs"]["input_ids"]

        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if self.config['position_embeddings']['mode'] == 'node_types':
            structural_position_ids = batch['node_types_labels']
            inputs_embeds += self.structural_position_embeddings(structural_position_ids)
        elif self.config['position_embeddings']['mode'] == 'node_depths':
            structural_position_ids = batch['node_depths_labels']
            inputs_embeds += self.structural_position_embeddings(structural_position_ids)

        return inputs_embeds

    # noinspection PyUnresolvedReferences,PyTypeChecker
    def training_collate_fn(self, instances: List[BaseInstance]) -> Any:
        if isinstance(instances[0], S2ORCITGSubsetInstance):
            return collate_s2orcitgsubset_instances(instances, self.tokenizer, self.config)

        text_and_spans = [to_input_sequence(instance.document, self.config) for instance in instances]
        texts = [text for text, _ in text_and_spans]
        node_spans = [spans for _, spans in text_and_spans]
        prompts = [instance.prompt for instance in instances]
        offsets = [len(instance.prompt) + 6 for instance in instances]
        dummy_labels = ['</s>' for _ in instances]
        label_codes = [instance.label_codes[0] for instance in instances]
        input_texts = [f"{prompt} </s> {text}" for prompt, text in zip(prompts, texts)]

        longt5_inputs = self.tokenizer(
            input_texts,
            text_target=dummy_labels,
            return_tensors="pt",
            padding="longest",
            truncation="longest_first",
            max_length=self.config["max_input_length"]
        )

        tokenized_structure = make_tokenized_structure(
            instances=instances,
            node_spans=node_spans,
            inputs=longt5_inputs,
            offsets=offsets,
            config=self.config
        )

        scaffold_tasks_labels_and_mask = make_scaffold_tasks_labels_and_mask(
            tokenized_structure=tokenized_structure,
            config=self.config
        )

        # create evidence labels and mask
        evidence_labels = []
        evidence_nodes = []
        evidence_mask = torch.zeros_like(longt5_inputs["input_ids"], dtype=torch.bool)

        for batch_ix, (span_mapping, instance) in enumerate(zip(node_spans, instances)):
            evidence_node_ids = set(node.ix for node in instance.evidence_nodes[0])
            offset = len(instance.prompt) + 6
            evidence_nodes_for_instance = []
            evidence_labels_for_instance = []
            for span in span_mapping.spans:
                # put label on first token in node
                token_ix = longt5_inputs.char_to_token(batch_ix, span.left + offset)
                if token_ix is not None:  # it's sometimes none because of truncation
                    if span.content.ntype in self.config["model"]["evidence"]["relevant_node_types"]:
                        evidence_mask[batch_ix, token_ix] = True
                        evidence_nodes_for_instance.append(span.content)
                        if span.content.ix in evidence_node_ids:
                            evidence_labels_for_instance.append(1)
                        else:
                            evidence_labels_for_instance.append(0)

            evidence_nodes.append(evidence_nodes_for_instance)
            evidence_labels.append(evidence_labels_for_instance)

        classification_labels = torch.tensor([
            label_code + 1 for label_code in label_codes  # map -1, 0, 1 to 0, 1, 2
        ], dtype=torch.long)

        return {
            "instances": instances,
            "node_spans": node_spans,
            "longt5_inputs": longt5_inputs,
            "classification_labels": classification_labels,
            "evidence_mask": evidence_mask,
            "evidence_labels": evidence_labels,
            "evidence_nodes": evidence_nodes,
            **tokenized_structure,
            **scaffold_tasks_labels_and_mask
        }

    def validation_collate_fn(self, instances: List[BaseInstance]) -> Any:
        return self.training_collate_fn(instances)


class LongT5ForScaffoldTaskForPretrainingModel(BaseModel):
    task_name: ClassVar[str] = 'Scaffold Task For Pretraining'
    model_name: ClassVar[str] = "LongT5"

    def __init__(self, config: omegaconf.DictConfig, stats: Statistics):
        super().__init__(config, stats)
        # set up tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/long-t5-tglobal-base"
        )
        self.tokenizer.add_tokens(get_structural_tokens(self.config))

        # set up model
        self.model = transformers.LongT5ForConditionalGeneration.from_pretrained(
            "google/long-t5-tglobal-base",
            use_cache=self.config["use_cache"]
        )

        num_structural_tokens = len(get_structural_tokens(self.config))
        self.model.resize_token_embeddings(self.model.config.vocab_size + num_structural_tokens)

        # prepare position embeddings
        self.batch_injector = {}
        if self.config["position_embeddings"]["mode"] != "vanilla":
            logger.info("Using structural position embeddings.")
            # noinspection PyTypeChecker
            self.structural_position_embeddings = LongT5StructuralPositionEmbedding(
                self.config,
                self.model.config.d_model
            )

        # prepare scaffolds
        self.scaffold_tasks_head = ScaffoldTasksHead(
            hidden_dimension=self.model.config.d_model,
            config=self.config
        )

        # Get parameters for t5 denoising
        self.dynamic_noise_span_length = config['model']['dynamic_noise_span_length']
        self.noise_density = config['model']['noise_density']
        self.mean_noise_span_length = config['model']['mean_noise_span_length']
        if config['max_input_length'] is None:
            self.max_input_length = 16384
        else:
            self.max_input_length = config['max_input_length']
        # Get number of sentinel tokens
        # * 2 to be sure there are enough sentinel tokens
        n_sentinel_tokens = 2 * get_n_noise_spans(
            self.max_input_length,
            self.noise_density,
            self.mean_noise_span_length
        )
        # Define sentinel tokens
        sentinel_tokens = [
            f'<sentinel_{i}>'
            for i in range(n_sentinel_tokens)
        ]
        self.sentinel_start_id = self.model.config.vocab_size
        # Add sentinel tokens
        self.tokenizer.add_tokens(sentinel_tokens)
        self.model.resize_token_embeddings(self.model.config.vocab_size + n_sentinel_tokens)

    def training_step(self, batch, batch_idx):
        self.batch_injector["batch"] = batch

        inputs_embeds = self.embed(batch)

        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch['inputs']["attention_mask"],
            labels=batch['inputs']["labels"],
        )

        loss = output.loss
        if loss is None:
            loss = 0

        all_hidden_states = output.encoder_last_hidden_state
        if self.config['scaffold_tasks']['mode'] != 'vanilla':
            scaffold_loss = self.scaffold_tasks_head(
                all_hidden_states=all_hidden_states,
                scaffold_tasks_mask=batch["scaffold_tasks_mask"],
                scaffold_tasks_node_types_labels=batch["scaffold_tasks_node_types_labels"],
                scaffold_tasks_node_depths_labels=batch["scaffold_tasks_node_depths_labels"]
            )
            if loss > 0:
                loss *= self.config['model']['denoising_loss_weight']
                loss += (1-self.config['model']['denoising_loss_weight']) * scaffold_loss

            else:
                loss = scaffold_loss

        return loss

    def validation_step(self, batch, batch_idx):
        self.batch_injector["batch"] = batch

        inputs_embeds = self.embed(batch)

        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch['inputs']["attention_mask"],
            labels=batch['inputs']["labels"],
        )

        loss = output.loss

        if loss is None:
            loss = 0

        all_hidden_states = output.encoder_last_hidden_state
        if self.config['scaffold_tasks']['mode'] != 'vanilla':
            scaffold_loss = self.scaffold_tasks_head(
                all_hidden_states=all_hidden_states,
                scaffold_tasks_mask=batch["scaffold_tasks_mask"],
                scaffold_tasks_node_types_labels=batch["scaffold_tasks_node_types_labels"],
                scaffold_tasks_node_depths_labels=batch["scaffold_tasks_node_depths_labels"]
            )
            if loss > 0:
                loss *= self.config['model']['denoising_loss_weight']
                loss += (1-self.config['model']['denoising_loss_weight']) * scaffold_loss

            else:
                loss = scaffold_loss

        # HACK: Duplicate the loss for each instance to return the same number
        # of predictions as instances
        predictions = [
            ScaffoldTaskForPreTrainingPrediction(loss.item())
            for _ in batch['instances']
        ]

        self.log('val_loss', float(loss.item()))

        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["model"]["learning_rate"]
        )
        if self.config['model']['lr_scheduler'] is None:
            return optimizer
        elif self.config['model']['lr_scheduler'] == 'linear':
            n_steps = self.config['model']['max_steps']
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1,
                end_factor=0,
                total_iters=n_steps
            )
            lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1
            }
            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler_config
            }
        else:
            raise ValueError(f"Unknown lr_scheduler {self.config['model']['lr_scheduler']}")

    def embed(
            self,
            batch: Dict[str, Any],
    ) -> torch.Tensor:
        input_ids = batch["inputs"]["input_ids"]

        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if self.config['position_embeddings']['mode'] == 'node_types':
            structural_position_ids = batch['node_types_labels']
            inputs_embeds += self.structural_position_embeddings(structural_position_ids)
        elif self.config['position_embeddings']['mode'] == 'node_depths':
            structural_position_ids = batch['node_depths_labels']
            inputs_embeds += self.structural_position_embeddings(structural_position_ids)

        return inputs_embeds

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Union[Optimizer, LightningOptimizer],
            optimizer_idx: int = 0,
            optimizer_closure: Optional[Callable[[], Any]] = None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False,
    ) -> None:
        optimizer.step(closure=optimizer_closure)

        if self.trainer.global_step < self.config['model']['warmup_steps']:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1) / self.config['model']['warmup_steps']
            )
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.config['model']['learning_rate']


    def training_collate_fn(self, instances: List[BaseInstance]) -> Any:

        # Copied from structformer.scaffold_tasks.collate_s2orcitgsubset_instances
        text_and_spans = [to_input_sequence(instance.document, self.config) for instance in instances]
        texts = [text for text, _ in text_and_spans]
        node_spans = [spans for _, spans in text_and_spans]
        offsets = [0 for _ in text_and_spans]

        initial_max_length = self.max_input_length

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation="longest_first",
            max_length=initial_max_length
        )

        if self.config['model']['do_denoising_pretraining']:
            # do masking here
            new_inputs = t5_mask_input_and_get_output(
                inputs,
                self.noise_density,
                self.mean_noise_span_length,
                self.sentinel_start_id,
                self.tokenizer.eos_token_id,
                self.config['max_input_length'],
                self.config['model']['target_length'],
                self.tokenizer.pad_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer,
                self.config['model']['dynamic_noise_span_length']
            )

            del inputs
            inputs = new_inputs
            del new_inputs

            inputs.data_to_torch()

        else:
            # Dummy input for decoder
            inputs["decoder_input_ids"] = torch.tensor([
                [0]
                for _ in inputs["input_ids"]
            ])

        tokenized_structure = make_tokenized_structure(
            instances=instances,
            node_spans=node_spans,
            inputs=inputs,
            offsets=offsets,
            config=self.config
        )

        scaffold_tasks_labels_and_mask = make_scaffold_tasks_labels_and_mask(
            tokenized_structure=tokenized_structure,
            config=self.config
        )

        post_encoder_position_embedding_ids = get_post_encoder_position_embedding_ids(
            self.config['post_encoder_position_embeddings']['mode'],
            tokenized_structure
        )

        inputs_dict = {
            'input_ids': inputs['input_ids'],
            'labels': inputs['labels'],
            'attention_mask': inputs['attention_mask'],
        }
        del inputs

        return {
            "is_s2orc_itg_subset_batch": False,
            "instances": instances,
            "node_spans": node_spans,
            "inputs": inputs_dict,
            "post_encoder_position_embedding_ids": post_encoder_position_embedding_ids,
            **tokenized_structure,
            **scaffold_tasks_labels_and_mask
        }

    def validation_collate_fn(self, instances: List[BaseInstance]) -> Any:
        return self.training_collate_fn(instances)