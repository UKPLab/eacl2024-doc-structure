"""Code for creating the global attention mask based on the tokenized input.

The functionalities in this module should be accessed via the public method:

- make_global_attention_mask
"""
import logging
from typing import List
import re

try:
    from omegaconf import DictConfig
except (ModuleNotFoundError, ImportError):
    from typing import Dict as DictConfig
import torch
from intertext_graph.itgraph import IntertextDocument, Node, Edge, Etype
from transformers import BatchEncoding, LEDTokenizerFast

from evaluation.common import BaseInstance
from structformer.input_sequence import get_structural_tokens, to_input_sequence
from structformer.sequence_alignment import SpanMapping

logger = logging.getLogger(__name__)


def _make_vanilla_global_attention_mask(
        instances: List[BaseInstance],
        node_spans: List[SpanMapping],
        inputs: BatchEncoding,
        offsets: List[int],
        tokenizer: LEDTokenizerFast,
        config: DictConfig
) -> None:
    """Create the vanilla task-specific global attention mask.

    Create the global attention mask that fits the given task. For example, put
    global attention on all question tokens for question answering.

    This method requires the following configuration keys:

    - model_wrapper/task_name

    Args:
        instances: The instances in the batch.
        node_spans: Mapping between characters and nodes.
        inputs: BatchEncoding object with the given inputs.
        offsets: Character offsets of the question/prompt before the document starts.
        tokenizer: Tokenizer used to tokenize the inputs.
        config: Complete Hydra configuration.
    """
    # we cannot import S2ORCITGSubsetInstance without causing circular imports,
    # so instead we have to check for the attribute
    if hasattr(instances[0], "is_s2orc_itg_subset_instance"):
        inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
        inputs["global_attention_mask"][:, 0] = 1
    else:
        task_name = config["model"]["task_name"]
        inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
        if task_name in ["QASPER", "Evidence Inference"]:
            close_id = tokenizer.convert_tokens_to_ids("</s>")
            idx = inputs["input_ids"][0].tolist().index(close_id)  # this could be faster...
            inputs["global_attention_mask"][:, :idx + 1] = 1
            # TODO: this currently only works for batch size 1
        elif task_name == 'Scaffold Task For Pretraining':
            inputs['global_attention_mask'][:, 0] = 1
        else:
            logger.error(f"Unknown global attention task '{task_name}'!")
            assert False, f"Unknown global attention task '{task_name}'!"


def _make_global_attention_on_structural_tokens_mask(
        instances: List[BaseInstance],
        node_spans: List[SpanMapping],
        inputs: BatchEncoding,
        offsets: List[int],
        tokenizer: LEDTokenizerFast,
        config: DictConfig,
        node_subsets: List[List[Node]] = None
) -> None:
    """Create the global attention mask with global attention on structural tokens.

    Create the global attention mask that fits the given task and has additional
    global attention on all structural tokens.

    Args:
        instances: The instances in the batch.
        node_spans: Mapping between characters and nodes.
        inputs: BatchEncoding object with the given inputs.
        offsets: Character offsets of the question/prompt before the document starts.
        tokenizer: Tokenizer used to tokenize the inputs.
        config: Complete Hydra configuration.
        node_subsets: One list of nodes per examples in batch.
            Only add global attention to these nodes.
    """
    _make_vanilla_global_attention_mask(instances, node_spans, inputs, offsets, tokenizer, config)

    # add global attention on structural tokens
    structural_tokens = get_structural_tokens(config)
    structural_token_ids = tokenizer.convert_tokens_to_ids(structural_tokens)
    structural_token_ids = set(structural_token_ids)
    for batch_ix, input_ids in enumerate(inputs["input_ids"]):
        for token_ix, input_id in enumerate(input_ids):
            if int(input_id) in structural_token_ids:
                if node_subsets is not None:
                    # If node subsets are provided, check whether
                    # current token is in allowed list of nodes
                    char_idx = inputs.token_to_chars(batch_ix, token_ix)
                    left_char = char_idx[0] - offsets[batch_ix]
                    for span in node_spans[batch_ix].spans:
                        if left_char >= span.left and left_char < span.right:
                            node_span = span
                    if node_span.content in node_subsets[batch_ix]:
                        inputs["global_attention_mask"][batch_ix, token_ix] = 1

                else:
                    inputs["global_attention_mask"][batch_ix, token_ix] = 1


def _make_global_attention_on_first_tokens_mask(
        instances: List[BaseInstance],
        node_spans: List[SpanMapping],
        inputs: BatchEncoding,
        offsets: List[int],
        tokenizer: LEDTokenizerFast,
        config: DictConfig,
        node_subsets: List[List[Node]] = None
) -> None:
    """Create the global attention mask with global attention on the first token of each node.

    Create the global attention mask that fits the given task and has additional
    global attention on the first token of each node.

    Args:
        instances: The instances in the batch.
        node_spans: Mapping between characters and nodes.
        inputs: BatchEncoding object with the given inputs.
        offsets: Character offsets of the question/prompt before the document starts.
        tokenizer: Tokenizer used to tokenize the inputs.
        config: Complete Hydra configuration.
        node_subsets: One list of nodes per examples in batch.
            Only add global attention to these nodes.
    """
    _make_vanilla_global_attention_mask(instances, node_spans, inputs, offsets, tokenizer, config)

    # add global attention on the first token of every node
    for batch_ix, (span_mapping, offset) in enumerate(zip(node_spans, offsets)):
        for span in span_mapping.spans:
            # put label on first token in node

            if node_subsets is not None:
                # If node subsets are provided, check whether
                # current token is in allowed list of nodes
                if span.content not in node_subsets[batch_ix]:
                    continue

            token_ix = inputs.char_to_token(batch_ix, span.left + offset)

            if (token_ix is not None):   # it's sometimes none because of truncation

                token = inputs[batch_ix].tokens[token_ix]
                closing_token_pattern = r'^</.+>'
                if re.findall(closing_token_pattern, token):
                    # skip this span when it starts with a closing token
                    continue

                sequence_id = inputs[batch_ix].sequence_ids[token_ix]
                if not sequence_id == 0:
                    continue

                inputs["global_attention_mask"][batch_ix, token_ix] = 1


def make_global_attention_mask(
        instances: List[BaseInstance],
        node_spans: List[SpanMapping],
        inputs: BatchEncoding,
        offsets: List[int],
        tokenizer: LEDTokenizerFast,
        config: DictConfig,
        node_subsets: List[List[Node]] = None
) -> None:
    """Create the global attention mask for the given inputs.

    This method does not return the global attention mask, but instead changes the
     given input's tensor "global_attention_mask".

    This method requires the following configuration keys:

    - attention/mode

    Args:
        instances: The instances in the batch.
        node_spans: Mapping between characters and nodes.
        inputs: BatchEncoding object with the given inputs.
        offsets: Character offsets of the question/prompt before the document starts.
        tokenizer: Tokenizer used to tokenize the inputs.
        config: Complete Hydra configuration.
        node_subsets: One list of nodes per examples in batch.
            Only add global attention to these nodes.
    """
    mode = config["attention"]["mode"]

    if mode == "vanilla":
        return _make_vanilla_global_attention_mask(instances, node_spans, inputs, offsets, tokenizer, config)
    elif mode == "global_attention_on_structural_tokens":
        return _make_global_attention_on_structural_tokens_mask(instances, node_spans, inputs, offsets, tokenizer, config, node_subsets)
    elif mode == "global_attention_on_first_tokens":
        return _make_global_attention_on_first_tokens_mask(instances, node_spans, inputs, offsets, tokenizer, config, node_subsets)
    else:
        logger.error(f"Unknown global attention mode '{mode}'!")
        assert False, f"Unknown global attention mode '{mode}'!"


if __name__ == "__main__":
    from omegaconf import OmegaConf

    doc = IntertextDocument([], [], "doc")

    title_node = Node("Some Great Title", ntype="article-title")
    doc.add_node(title_node)

    abstract_node = Node("Abstract", ntype="abstract")
    doc.add_node(abstract_node)
    doc.add_edge(Edge(title_node, abstract_node, Etype.PARENT))
    doc.add_edge(Edge(title_node, abstract_node, Etype.NEXT))

    abstract_paragraph_node = Node("This is a concise abstract.", ntype="p")
    doc.add_node(abstract_paragraph_node)
    doc.add_edge(Edge(abstract_node, abstract_paragraph_node, Etype.PARENT))
    doc.add_edge(Edge(abstract_node, abstract_paragraph_node, Etype.NEXT))

    section_node = Node("A Descriptive Section Title", ntype="title")
    doc.add_node(section_node)
    doc.add_edge(Edge(title_node, section_node, Etype.PARENT))
    doc.add_edge(Edge(abstract_paragraph_node, section_node, Etype.NEXT))

    paragraph_node_1 = Node("An interesting first paragraph.", ntype="p")
    doc.add_node(paragraph_node_1)
    doc.add_edge(Edge(section_node, paragraph_node_1, Etype.PARENT))
    doc.add_edge(Edge(section_node, paragraph_node_1, Etype.NEXT))

    paragraph_node_2 = Node("An interesting second paragraph.", ntype="p")
    doc.add_node(paragraph_node_2)
    doc.add_edge(Edge(section_node, paragraph_node_2, Etype.PARENT))
    doc.add_edge(Edge(paragraph_node_1, paragraph_node_2, Etype.NEXT))

    print("Document Plaintext:")
    print(doc.to_plaintext())

    print("\n" * 4)
    config = OmegaConf.create({
        "model": {
            "task_name": "QASPER"
        },
        "input_sequence": {
            "mode": "text_with_node_types",
            "node_separator": " ",
            "do_close": True,
            "replace_newlines": False,
            "include_node_types": ["article-title", "abstract", "title", "p"]
        },
        "attention": {
            "mode": "global_attention_on_structural_tokens",
            "task_category": "question_answering"
        }
    })

    text, node_spans = to_input_sequence(doc, config)
    question = "How interesting is the paper?"

    input_text = f"{question} </s> {text}"
    offsets = [len(question) + 6]
    print(input_text)

    tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
    tokenizer.add_tokens(get_structural_tokens(config))

    inputs = tokenizer(
        [input_text],
        return_tensors="pt",
        padding="longest",
        truncation="longest_first",
        max_length=16843
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(tokens)

    make_global_attention_mask([None], [node_spans], inputs, offsets, tokenizer, config)

    print(inputs["global_attention_mask"])
    print(torch.sum(inputs["global_attention_mask"]))