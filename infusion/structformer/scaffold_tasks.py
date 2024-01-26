import dataclasses
import glob
import logging
import os
import time
from io import StringIO
from typing import List, Dict

import omegaconf
import torch
import torch.nn.functional as F
from intertext_graph.itgraph import IntertextDocument
from torch import nn
from transformers import LEDTokenizerFast

from evaluation.common import BaseInstance, Statistics
from structformer.global_attention import make_global_attention_mask
from structformer.input_sequence import to_input_sequence
from structformer.tokenized_structure import make_tokenized_structure

logger = logging.getLogger(__name__)


def make_scaffold_tasks_labels_and_mask(
        tokenized_structure: Dict[str, torch.Tensor],
        config: omegaconf.DictConfig
):
    token_chance = config["scaffold_tasks"]["token_chance"]

    node_types_mask = tokenized_structure["node_types_mask"]
    node_types_labels = tokenized_structure["node_types_labels"]
    node_depths_mask = tokenized_structure["node_depths_mask"]
    node_depths_labels = tokenized_structure["node_depths_labels"]

    scaffold_tasks_mask = torch.zeros_like(node_types_mask, dtype=torch.bool)
    scaffold_tasks_node_types_labels = []
    scaffold_tasks_node_depths_labels = []

    random_draws = torch.rand(scaffold_tasks_mask.shape)

    for batch_ix in range(scaffold_tasks_mask.shape[0]):
        for token_ix in range(scaffold_tasks_mask.shape[1]):
            if float(random_draws[batch_ix, token_ix]) < token_chance:
                if node_types_mask[batch_ix, token_ix] and node_depths_mask[batch_ix, token_ix]:
                    scaffold_tasks_mask[batch_ix, token_ix] = True
                    scaffold_tasks_node_types_labels.append(int(node_types_labels[batch_ix, token_ix]))
                    scaffold_tasks_node_depths_labels.append(int(node_depths_labels[batch_ix, token_ix]))

    scaffold_tasks_node_types_labels = torch.tensor(scaffold_tasks_node_types_labels, dtype=torch.long)
    scaffold_tasks_node_depths_labels = torch.tensor(scaffold_tasks_node_depths_labels, dtype=torch.long)

    return {
        "scaffold_tasks_mask": scaffold_tasks_mask,
        "scaffold_tasks_node_types_labels": scaffold_tasks_node_types_labels,
        "scaffold_tasks_node_depths_labels": scaffold_tasks_node_depths_labels
    }


class ScaffoldTasksHead(nn.Module):

    def __init__(self, hidden_dimension: int, config: omegaconf.DictConfig) -> None:
        super(ScaffoldTasksHead, self).__init__()
        self.config = config

        if self.config["scaffold_tasks"]["mode"] in ["node_types", "node_types_and_node_depths"]:
            self.node_types_linear_layer_1 = torch.nn.Linear(
                in_features=hidden_dimension,
                out_features=hidden_dimension
            )
            self.node_types_linear_layer_2 = torch.nn.Linear(
                in_features=hidden_dimension,
                out_features=len(self.config["node_types"]) + 1
            )

        if self.config["scaffold_tasks"]["mode"] in ["node_depths", "node_types_and_node_depths"]:
            self.node_depths_linear_layer_1 = torch.nn.Linear(
                in_features=hidden_dimension,
                out_features=hidden_dimension
            )
            self.node_depths_linear_layer_2 = torch.nn.Linear(
                in_features=hidden_dimension,
                out_features=self.config["max_depth"] + 1
            )

    def forward(
            self,
            all_hidden_states: torch.Tensor,
            scaffold_tasks_mask: torch.Tensor,
            scaffold_tasks_node_types_labels: torch.Tensor,
            scaffold_tasks_node_depths_labels: torch.Tensor
    ) -> torch.Tensor:
        mode = self.config["scaffold_tasks"]["mode"]

        if mode == "vanilla":
            logger.error("The scaffold tasks head should never be called with mode 'vanilla'!")
            assert False, "The scaffold tasks head should never be called with mode 'vanilla'!"

        if mode not in ["node_types", "node_depths", "node_types_and_node_depths"]:
            logger.error(f"Unknown scaffold tasks mode '{mode}'!")
            assert False, f"Unknown scaffold tasks mode '{mode}'!"

        relevant_hidden_states = all_hidden_states[scaffold_tasks_mask]

        node_types_loss = None
        if mode in ["node_types", "node_types_and_node_depths"]:
            tmp = self.node_types_linear_layer_1(relevant_hidden_states)
            tmp = torch.tanh(tmp)
            node_types_logits = self.node_types_linear_layer_2(tmp)

            node_types_loss = F.cross_entropy(
                node_types_logits,
                scaffold_tasks_node_types_labels
            )

            if mode == "node_types":
                return node_types_loss

        node_depths_loss = None
        if mode in ["node_depths", "node_types_and_node_depths"]:
            tmp = self.node_depths_linear_layer_1(relevant_hidden_states)
            tmp = torch.tanh(tmp)
            node_depths_logits = self.node_depths_linear_layer_2(tmp)

            node_depths_loss = F.cross_entropy(
                node_depths_logits,
                scaffold_tasks_node_depths_labels
            )

            if mode == "node_depths":
                return node_depths_loss

        return node_types_loss + node_depths_loss


@dataclasses.dataclass
class S2ORCITGSubsetInstance(BaseInstance):
    # input:
    document: IntertextDocument

    # this is necessary to avoid circular imports:
    # global_attention cannot import scaffold_task but must discern
    # S2ORCITGSubsetInstance instances.
    is_s2orc_itg_subset_instance: bool = True


class S2ORCITGSubsetDataHandler:
    _instances: List[S2ORCITGSubsetInstance]

    def __init__(self, config: omegaconf.DictConfig, stats: Statistics) -> None:
        super(S2ORCITGSubsetDataHandler, self).__init__()
        self.config = config
        self.stats = stats

        logger.info("Load the S2ORC-ITG-Subset.")
        tick = time.time()

        path = os.path.join(self.config["location"]["datasets"], "S2ORC-ITG-Subset", "shards", "*.jsonl")
        paths = glob.glob(path)
        paths.sort()

        documents = []
        num_docs_per_shard = self.config["scaffold_tasks"]["num_docs_per_shard"]
        for path in paths:
            with open(path, "r", encoding="utf-8") as file:
                n = 0
                for line in file:
                    with StringIO(line) as f:
                        document = IntertextDocument.load_json(f)
                        documents.append(document)
                    n += 1
                    if n == num_docs_per_shard:
                        break

        self.stats.stats["s2orc-itg-subset-initialization.num-documents"] = len(documents)

        tack = time.time()
        logger.info(f"Loaded {len(documents)} documents in {tack - tick:0.4f}s.")

        logger.info("Create the S2ORC-ITG-Subset instances.")
        tick = time.time()

        self._instances = []
        for document in documents:
            instance = S2ORCITGSubsetInstance(
                document=document
            )
            self._instances.append(instance)

        self.stats.stats["s2orc-itg-subset-initialization.num-train-instances"] = len(self.instances)

        tack = time.time()
        logger.info(f"Created {len(self.instances)} instances in {tack - tick:0.4f}s.")

    @property
    def instances(self) -> List[S2ORCITGSubsetInstance]:
        return self._instances


def collate_s2orcitgsubset_instances(
        instances: List[S2ORCITGSubsetInstance],
        tokenizer: LEDTokenizerFast,
        config: omegaconf.DictConfig
):
    text_and_spans = [to_input_sequence(instance.document, config) for instance in instances]
    texts = [text for text, _ in text_and_spans]
    node_spans = [spans for _, spans in text_and_spans]
    offsets = [0 for _ in text_and_spans]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="longest",
        truncation="longest_first",
        max_length=config["max_input_length"]
    )

    make_global_attention_mask(
        instances=instances,
        node_spans=node_spans,
        inputs=inputs,
        offsets=offsets,
        tokenizer=tokenizer,
        config=config
    )

    tokenized_structure = make_tokenized_structure(
        instances=instances,
        node_spans=node_spans,
        inputs=inputs,
        offsets=offsets,
        config=config
    )

    scaffold_tasks_labels_and_mask = make_scaffold_tasks_labels_and_mask(
        tokenized_structure=tokenized_structure,
        config=config
    )

    return {
        "is_s2orc_itg_subset_batch": True,
        "instances": instances,
        "node_spans": node_spans,
        "inputs": inputs,
        **tokenized_structure,
        **scaffold_tasks_labels_and_mask
    }
