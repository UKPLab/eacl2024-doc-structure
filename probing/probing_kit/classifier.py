from typing import Dict

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.nn.util import get_text_field_mask, get_token_ids_from_text_field_tensors
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
import numpy as np
import torch
import torch.nn.functional as F


class ProbingModel(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            token_embedder: PretrainedTransformerEmbedder,
            span_extractor: EndpointSpanExtractor,
            num_spans: int,
            batch_injector: Dict = None) -> None:
        super().__init__(vocab)
        self._token_embedder = token_embedder
        self._text_field_embedder = BasicTextFieldEmbedder({'tokens': token_embedder})
        self._span_extractor = span_extractor
        self._classifier_input_dim = self._span_extractor.get_output_dim() * num_spans
        num_labels = vocab.get_vocab_size('labels')
        self.num_labels = num_labels
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self.num_labels)
        self._metrics = {
            'accuracy': CategoricalAccuracy(),
            'fbeta': FBetaMeasure(average='micro')
        }
        self.batch_injector = batch_injector

    def forward(
            self,
            tokens: TextFieldTensors,
            labels: torch.Tensor,
            spans_1: torch.Tensor,
            spans_2: torch.Tensor = None,
            node_types_ids: torch.Tensor = None,
            node_depths_ids: torch.Tensor = None,
            global_attention_mask: torch.Tensor = None
            ) -> Dict[str, torch.Tensor]:
        # Set batch injector (is accessed in token_embedder)
        self.batch_injector['node_types_ids'] = node_types_ids
        self.batch_injector['node_depths_ids'] = node_depths_ids
        self.batch_injector['global_attention_mask'] = global_attention_mask
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_tokens = self._text_field_embedder(tokens)
        # Shape: (batch_size, num_tokens)
        mask = get_text_field_mask(tokens)
        # Shape: (batch_size, sequence_len * num_spans, 2)
        span_indices = torch.concat((spans_1, spans_2), dim=1) if spans_2 is not None else spans_1
        # Shape: (batch_size, sequence_len * num_spans)
        span_indices_mask = (span_indices[:, :, 0] >= 0).long()
        # Shape: (batch_size, sequence_len * num_spans, embedding_dim * num_combinations)
        embedded_spans = self._span_extractor(embedded_tokens, span_indices, mask, span_indices_mask)
        if spans_2 is not None:
            # Shape: (batch_size, sequence_len, embedding_dim * num_combinations * num_spans)
            embedded_spans = torch.cat(embedded_spans.chunk(2, dim=1), dim=-1)
        # Shape: (batch_size, sequence_len, num_labels)
        logits = self._classification_layer(embedded_spans)
        # Shape: (batch_size, sequence_len, num_labels)
        probs = F.softmax(logits, dim=-1)
        labels = labels.view(-1, 1)
        labels_mask = labels != -1
        # Shape: (batch_size * sequence_len,)
        labels = torch.masked_select(labels, labels_mask)
        logits = logits.view(-1, self.num_labels)
        # Shape: (batch_size * sequence_len, num_labels)
        logits = torch.masked_select(logits, labels_mask).view(-1, self.num_labels)
        # If there is one label do regression
        output_dict = {'logits': logits, 'probs': probs, 'token_ids': get_token_ids_from_text_field_tensors(tokens)}
        if labels is not None:
            # Shape: (1,)
            output_dict['loss'] = F.cross_entropy(logits, labels)
            for metric in self._metrics.values():
                metric(logits, labels)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for key, metric in self._metrics.items():
            if key == 'fbeta':
                metrics.update(**metric.get_metric(reset))
            else:
                metrics[key] = metric.get_metric(reset)
        return metrics

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = output_dict['logits'].cpu().data.numpy()
        predicted_id = np.argmax(logits, axis=-1)
        output_dict['labels'] = [self.vocab.get_token_from_index(x, namespace='labels') for x in predicted_id]  # type: ignore
        return output_dict


class AtomicModel(ProbingModel):

    def _embed_span(
            self,
            tokens: TextFieldTensors,
            span_indices: torch.Tensor,
            node_types_ids: torch.Tensor = None,
            node_depths_ids: torch.Tensor = None,
            global_attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        self.batch_injector.update({
            'node_types_ids': node_types_ids,
            'node_depths_ids': node_depths_ids,
            'global_attention_mask': global_attention_mask
        })
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_tokens = self._text_field_embedder(tokens)
        # Shape: (batch_size, num_tokens)
        mask = get_text_field_mask(tokens)
        # Shape: (batch_size, embedding_dim * num_combinations)
        return self._span_extractor(embedded_tokens, span_indices, mask)

    def forward(
            self,
            tokens_1: TextFieldTensors,
            label: torch.Tensor,
            span_1: torch.Tensor,
            tokens_2: TextFieldTensors = None,
            span_2: torch.Tensor = None,
            node_types_ids_1: torch.Tensor = None,
            node_depths_ids_1: torch.Tensor = None,
            global_attention_mask_1: torch.Tensor = None,
            node_types_ids_2: torch.Tensor = None,
            node_depths_ids_2: torch.Tensor = None,
            global_attention_mask_2: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, embedding_dim * num_combinations)
        embedded_spans = self._embed_span(
            tokens_1,
            span_1,
            node_types_ids_1,
            node_depths_ids_1,
            global_attention_mask_1
        )
        if tokens_2 is not None and span_2 is not None:
            # Shape: (batch_size, embedding_dim * num_combinations)
            tmp = self._embed_span(
                tokens_2,
                span_2,
                node_types_ids_2,
                node_depths_ids_2,
                global_attention_mask_2
            )
            # Shape: (batch_size, embedding_dim * num_combinations * num_spans)
            embedded_spans = torch.concat((embedded_spans, tmp), dim=-1)
        # Shape: (batch_size, num_labels)
        logits = self._classification_layer(embedded_spans)
        # Shape: (batch_size, num_labels)
        probs = F.softmax(logits, dim=-1)
        output_dict = {'logits': logits, 'probs': probs}
        if label is not None:
            # Shape: (1,)
            output_dict['loss'] = F.cross_entropy(logits, label)
            for metric in self._metrics.values():
                metric(logits, label)
        return output_dict
