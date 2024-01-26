import logging
from typing import List, Dict

try:
    from omegaconf import DictConfig
except (ModuleNotFoundError, ImportError):
    from typing import Dict as DictConfig
import torch
from intertext_graph.itgraph import Etype
from transformers import BatchEncoding

from evaluation.common import BaseInstance
from structformer.sequence_alignment import SpanMapping

logger = logging.getLogger(__name__)


def make_tokenized_structure(
        instances: List[BaseInstance],
        node_spans: List[SpanMapping],
        inputs: BatchEncoding,
        offsets: List[int],
        config: DictConfig
) -> Dict[str, torch.Tensor]:
    max_depth = config["max_depth"]
    node_types = config["node_types"]

    input_ids = inputs["input_ids"]

    node_types_map = {node_type: ix + 1 for ix, node_type in enumerate(node_types)}  # 0 used for prompt/unknown type...

    node_types_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    node_types_labels = torch.zeros_like(input_ids, dtype=torch.long)

    node_depths_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    node_depths_labels = torch.zeros_like(input_ids, dtype=torch.long)

    for batch_ix, (offset, span_mapping, instance) in enumerate(zip(offsets, node_spans, instances)):
        token_indices = []
        lefts = []
        # start at 1 because token_to_chars fails at index 0 (which is a special token)
        # stop at -1 because token_to_chars fails at index -1 (which is a special token)
        for token_ix in range(1, input_ids.shape[1] - 1):
            try:
                char_span = inputs.token_to_chars(batch_ix, token_ix)
                left = char_span.start - offset
                if (
                    left >= 0 # < 0 if token is not part of the ITG document
                    and inputs[batch_ix].sequence_ids[token_ix] == 0 # 1 if sentinel token
                ):
                    token_indices.append(token_ix)
                    lefts.append(left)

            except AttributeError:
                # token_to_chars fails for pad tokens
                continue

        # filter out the case in which the tokenizer puts a ' ' token between
        # the end of the document and the '</s>' token
        if lefts[-1] >= span_mapping.spans[-1].right:
            lefts = lefts[:-1]
        nodes = span_mapping.get_content_list(lefts)

        for token_ix, node in zip(token_indices, nodes):
            if node.ntype in node_types_map.keys():
                node_types_mask[batch_ix, token_ix] = True
                node_types_labels[batch_ix, token_ix] = node_types_map[node.ntype]

            depth = len(list(instance.document.breadcrumbs(node, Etype.PARENT)))
            assert 0 < depth
            if depth <= max_depth:
                node_depths_mask[batch_ix, token_ix] = True
                node_depths_labels[batch_ix, token_ix] = depth

    return {
        "node_types_mask": node_types_mask,
        "node_types_labels": node_types_labels,
        "node_depths_mask": node_depths_mask,
        "node_depths_labels": node_depths_labels
    }
