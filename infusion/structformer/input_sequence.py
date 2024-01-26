"""Code for transforming ITG documents into transformer input strings.

The functionalities in this module should be accessed via the two public methods:

- to_input_sequence
- get_structural_tokens
"""
import logging
from typing import Tuple, List

try:
    from omegaconf import DictConfig
except (ModuleNotFoundError, ImportError):
    from typing import Dict as DictConfig
from intertext_graph.itgraph import IntertextDocument, Etype, Node, Edge

from structformer.sequence_alignment import SpanMapping, Span

logger = logging.getLogger(__name__)


def _join_parts_and_create_span_mapping(
        parts: List[str],
        nodes: List[Node],
        separator: str
) -> Tuple[str, SpanMapping]:
    """Join the given list of string parts and create a mapping to the nodes.

    The separators will always be counted to the parts that precede them.

    Args:
        parts: Given list of string parts.
        nodes: Given list of corresponding nodes.
        separator: Separator string.

    Returns:
        The joined string parts and a mapping to the nodes.
    """
    spans = []
    current_left = 0
    node_sep_len = len(separator)
    for ix, (part, node) in enumerate(zip(parts, nodes)):
        right = current_left + len(part)
        if ix < len(parts) - 1:
            right += node_sep_len
        spans.append(Span(current_left, right, node))
        current_left = right

    return separator.join(parts), SpanMapping(spans)


################################################################################
# to text
################################################################################

_plain_text_structural_tokens = []


def _to_plain_text(
        document: IntertextDocument,
        config: DictConfig
) -> Tuple[str, SpanMapping]:
    """Transform the given ITG document into a plain text linear string.

    This method requires the following configuration keys:

    - input_sequence/include_node_types
    - input_sequence/replace_newlines
    - input_sequence/node_separator

    Args:
        document: ITG document.
        config: Complete Hydra configuration.

    Returns:
        Transformer input string, Node mapping.
    """
    include_node_types = config["input_sequence"]["include_node_types"]
    replace_newlines = config["input_sequence"]["replace_newlines"]
    node_separator = config["input_sequence"]["node_separator"]

    parts = []
    nodes = []
    for node in document.unroll_graph():
        if include_node_types is None or node.ntype in include_node_types:
            if replace_newlines:
                text = node.content.replace("\n", " ")
            else:
                text = node.content

            parts.append(text)
            nodes.append(node)

    return _join_parts_and_create_span_mapping(parts, nodes, node_separator)


################################################################################
# to text with node-boundary based structural tokens
################################################################################

_text_with_node_boundaries_structural_tokens = [
    "<node>"
]

_text_with_node_boundaries_structural_tokens_with_closing = [
    "<node>", "</node>"
]


def _to_text_with_node_boundaries(
        document: IntertextDocument,
        config: DictConfig
) -> Tuple[str, SpanMapping]:
    """Place structural tokens between the nodes.

    Transform the given ITG document into a linear string and put structural
    tokens between the nodes.

    This method requires the following configuration keys:

    - input_sequence/include_node_types
    - input_sequence/replace_newlines
    - input_sequence/node_separator
    - input_sequence/do_close
    - input_sequence/use_bos_eos_token
    - input_sequence/bos_token
    - input_sequence/eos_token

    Args:
        document: ITG document.
        config: Complete Hydra configuration.

    Returns:
        Transformer input string, Node mapping.
    """
    include_node_types = config["input_sequence"]["include_node_types"]
    replace_newlines = config["input_sequence"]["replace_newlines"]
    node_separator = config["input_sequence"]["node_separator"]
    do_close = config["input_sequence"]["do_close"]

    if config['input_sequence']['use_bos_eos_token']:
        structural_token = config['input_sequence']['bos_token']
        closing_token = config['input_sequence']['eos_token']
    else:
        structural_token = _text_with_node_boundaries_structural_tokens_with_closing[0]
        closing_token = _text_with_node_boundaries_structural_tokens_with_closing[1]

    parts, nodes = [], []
    part_stack, node_stack = [], []
    for node in document.unroll_graph():
        if include_node_types is None or node.ntype in include_node_types:
            # close nodes that need to be closed
            if do_close and node_stack != []:
                # find this node's parent in the node stack
                parent = list(document.breadcrumbs(node, Etype.PARENT))[1]
                idx = 0
                for ix, stack_node in enumerate(reversed(node_stack)):
                    if stack_node is parent:
                        idx = ix
                        break

                # close all nodes until the parent and remove them from stacks
                for ix in range(idx):
                    nodes.append(node_stack.pop())
                    parts.append(part_stack.pop())

            if replace_newlines:
                text = node.content.replace("\n", " ")
            else:
                text = node.content

            node_input_seq = f"{structural_token} {text}"
            parts.append(node_input_seq)
            nodes.append(node)

            if do_close:
                part_stack.append(closing_token)
                node_stack.append(node)

    # close remaining nodes
    if do_close and node_stack != []:
        for node, tag in zip(reversed(node_stack), reversed(part_stack)):
            parts.append(tag)
            nodes.append(node)

    return _join_parts_and_create_span_mapping(parts, nodes, node_separator)


################################################################################
# to text with node-type based structural tokens
################################################################################

# Keep separate lists of "core" structural tokens for basic node
# types to ensure compatibility to older models
_text_with_node_types_core_structural_tokens = [
    "<p>",
    "<title>",
    "<article-title>",
    "<abstract>"
]
_text_with_node_types_core_structural_tokens_with_closing = [
    "<p>", "</p>",
    "<title>", "</title>",
    "<article-title>", "</article-title>",
    "<abstract>", "</abstract>"
]

_text_with_node_types_structural_tokens = [
    "<p>",
    "<title>",
    "<article-title>",
    "<abstract>",
    "<list>",
    "<list-item>",
    "<table>",
    "<table-row>",
    "<table-item>",
]
_text_with_node_types_structural_tokens_with_closing = [
    "<p>", "</p>",
    "<title>", "</title>",
    "<article-title>", "</article-title>",
    "<abstract>", "</abstract>",
    "<list>", "</list>",
    "<list-item>", "</list-item>",
    "<table>", "</table>",
    "<table-row>", "</table-row>",
    "<table-item>", "</table-item>"
]

def _to_text_with_node_types(
        document: IntertextDocument,
        config: DictConfig
) -> Tuple[str, SpanMapping]:
    """Place structural tokens that describe the node types between nodes.

    Transform the given ITG document into a linear string and put structural
    tokens that encode the node types between the nodes.

    This method requires the following configuration keys:

    - input_sequence/include_node_types
    - input_sequence/replace_newlines
    - input_sequence/node_separator
    - input_sequence/do_close
    - input_sequence/use_core_node_types_only

    Args:
        document: ITG document.
        config: Complete Hydra configuration.

    Returns:
        Transformer input string, Node mapping.
    """
    include_node_types = config["input_sequence"]["include_node_types"]
    replace_newlines = config["input_sequence"]["replace_newlines"]
    node_separator = config["input_sequence"]["node_separator"]
    do_close = config["input_sequence"]["do_close"]
    use_core_node_types_only = config['input_sequence']['use_core_node_types_only']

    if use_core_node_types_only:
        if do_close:
            structural_tokens = _text_with_node_types_core_structural_tokens_with_closing
        else:
            structural_tokens = _text_with_node_types_core_structural_tokens

    else:
        if do_close:
            structural_tokens = _text_with_node_types_structural_tokens_with_closing
        else:
            structural_tokens = _text_with_node_types_structural_tokens

    parts, nodes = [], []
    part_stack, node_stack = [], []
    for node in document.unroll_graph():
        if include_node_types is None or node.ntype in include_node_types:
            # close nodes that need to be closed
            if do_close and node_stack != []:
                # find this node's parent in the node stack
                parent = list(document.breadcrumbs(node, Etype.PARENT))[1]
                idx = 0
                for ix, stack_node in enumerate(reversed(node_stack)):
                    if stack_node is parent:
                        idx = ix
                        break

                # close all nodes until the parent and remove them from stacks
                for ix in range(idx):
                    nodes.append(node_stack.pop())
                    parts.append(part_stack.pop())

            # add the current node
            structural_token = f"<{node.ntype}>"
            closing_token = f"</{node.ntype}>"
            if (structural_token not in structural_tokens):
                logger.error(f"Unknown structural token '{structural_token}'!")
                assert False, f"Unknown structural token '{structural_token}'!"

            if do_close and closing_token not in structural_tokens:
                logger.error(f"Unknown structural token '{closing_token}'!")
                assert False, f"Unknown structural token '{closing_token}'!"

            if replace_newlines:
                text = node.content.replace("\n", " ")
            else:
                text = node.content

            node_input_seq = f"{structural_token} {text}"
            parts.append(node_input_seq)
            nodes.append(node)

            if do_close:
                part_stack.append(closing_token)
                node_stack.append(node)

    # close remaining nodes
    if do_close and node_stack != []:
        for node, tag in zip(reversed(node_stack), reversed(part_stack)):
            parts.append(tag)
            nodes.append(node)

    return _join_parts_and_create_span_mapping(parts, nodes, node_separator)


################################################################################
# to text with node-depth based structural tokens
################################################################################

_text_with_node_depths_structural_tokens = [
    "<node-1>",
    "<node-2>",
    "<node-3>",
    "<node-4>",
    "<node-5>",
    "<node-6>",
    "<node-7>",
    "<node-8>",
    "<node-9>",
    "<node-10>",
    "<node-11>",
    "<node-12>",
    "<node-13>",
    "<node-14>",
    "<node-15>",
    "<node-16>",
    "<node-17>",
    "<node-18>",
    "<node-19>",
    "<node-20>"
]

_text_with_node_depths_structural_tokens_with_closing = [
    "<node-1>", "</node-1>",
    "<node-2>", "</node-2>",
    "<node-3>", "</node-3>",
    "<node-4>", "</node-4>",
    "<node-5>", "</node-5>",
    "<node-6>", "</node-6>",
    "<node-7>", "</node-7>",
    "<node-8>", "</node-8>",
    "<node-9>", "</node-9>",
    "<node-10>", "</node-10>",
    "<node-11>", "</node-11>",
    "<node-12>", "</node-12>",
    "<node-13>", "</node-13>",
    "<node-14>", "</node-14>",
    "<node-15>", "</node-15>",
    "<node-16>", "</node-16>",
    "<node-17>", "</node-17>",
    "<node-18>", "</node-18>",
    "<node-19>", "</node-19>",
    "<node-20>", "</node-20>"
]


def _to_text_with_node_depths(
        document: IntertextDocument,
        config: DictConfig
) -> Tuple[str, SpanMapping]:
    """Place structural tokens that describe the node depths between nodes.

    Transform the given ITG document into a linear string and put structural
    tokens that encode the node depths between the nodes.

    This method requires the following configuration keys:

    - input_sequence/include_node_types
    - input_sequence/replace_newlines
    - input_sequence/node_separator
    - input_sequence/do_close

    Args:
        document: ITG document.
        config: Complete Hydra configuration.

    Returns:
        Transformer input string, Node mapping.
    """
    include_node_types = config["input_sequence"]["include_node_types"]
    replace_newlines = config["input_sequence"]["replace_newlines"]
    node_separator = config["input_sequence"]["node_separator"]
    do_close = config["input_sequence"]["do_close"]

    parts, nodes = [], []
    part_stack, node_stack = [], []
    for node in document.unroll_graph():
        if include_node_types is None or node.ntype in include_node_types:
            # close nodes that need to be closed
            if do_close and node_stack != []:
                # find this node's parent in the node stack
                parent = list(document.breadcrumbs(node, Etype.PARENT))[1]
                idx = 0
                for ix, stack_node in enumerate(reversed(node_stack)):
                    if stack_node is parent:
                        idx = ix
                        break

                # close all nodes until the parent and remove them from stacks
                for ix in range(idx):
                    nodes.append(node_stack.pop())
                    parts.append(part_stack.pop())

            # add the current node
            depth = len(list(document.breadcrumbs(node, Etype.PARENT)))
            assert 0 < depth <= 20

            structural_token = f"<node-{depth}>"
            closing_token = f"</node-{depth}>"

            if (
                    (not do_close and structural_token not in _text_with_node_depths_structural_tokens)
                    or (do_close and structural_token not in _text_with_node_depths_structural_tokens_with_closing)
            ):
                logger.error(f"Unknown structural token '{structural_token}'!")
                assert False, f"Unknown structural token '{structural_token}'!"

            if do_close and closing_token not in _text_with_node_depths_structural_tokens_with_closing:
                logger.error(f"Unknown structural token '{closing_token}'!")
                assert False, f"Unknown structural token '{closing_token}'!"

            if replace_newlines:
                text = node.content.replace("\n", " ")
            else:
                text = node.content

            node_input_seq = f"{structural_token} {text}"
            parts.append(node_input_seq)
            nodes.append(node)

            if do_close:
                part_stack.append(closing_token)
                node_stack.append(node)

    # close remaining nodes
    if do_close and node_stack != []:
        for node, tag in zip(reversed(node_stack), reversed(part_stack)):
            parts.append(tag)
            nodes.append(node)

    return _join_parts_and_create_span_mapping(parts, nodes, node_separator)


################################################################################
# access
################################################################################


def to_input_sequence(
        document: IntertextDocument,
        config: DictConfig
) -> Tuple[str, SpanMapping]:
    """Transform the given ITG document into a linear string.

    This method requires the following configuration keys:

    - input_sequence/mode

    Args:
        document: ITG document.
        config: Complete Hydra configuration.

    Returns:
        Transformer input string, Node mapping.
    """
    mode = config["input_sequence"]["mode"]

    if mode == "vanilla":
        return _to_plain_text(document, config)
    elif mode == "text_with_node_boundaries":
        return _to_text_with_node_boundaries(document, config)
    elif mode == "text_with_node_types":
        return _to_text_with_node_types(document, config)
    elif mode == "text_with_node_depths":
        return _to_text_with_node_depths(document, config)
    else:
        logger.error(f"Unknown input sequence mode '{mode}'!")
        assert False, f"Unknown input sequence mode '{mode}'!"


def get_structural_tokens(config: DictConfig) -> List[str]:
    """Get the structural tokens for the given mode.

    This method requires the following configuration keys:

    - input_sequence/mode
    - input_sequence/do_close
    - input_sequence/use_core_node_types_only


    Args:
        config: Complete Hydra configuration.

    Returns:
        List of structural tokens.
    """
    mode = config["input_sequence"]["mode"]
    do_close = config["input_sequence"]["do_close"]
    use_core_node_types_only = config['input_sequence']['use_core_node_types_only']

    if mode == "vanilla":
        return _plain_text_structural_tokens
    elif mode == "text_with_node_boundaries":
        if do_close:
            return _text_with_node_boundaries_structural_tokens_with_closing
        else:
            return _text_with_node_boundaries_structural_tokens
    elif mode == "text_with_node_types":
        if use_core_node_types_only:
            if do_close:
                return _text_with_node_types_core_structural_tokens_with_closing
            else:
                return _text_with_node_types_core_structural_tokens
        else:
            if do_close:
                return _text_with_node_types_structural_tokens_with_closing
            else:
                return _text_with_node_types_structural_tokens
    elif mode == "text_with_node_depths":
        if do_close:
            return _text_with_node_depths_structural_tokens_with_closing
        else:
            return _text_with_node_depths_structural_tokens
    else:
        logger.error(f"Unknown input sequence mode '{mode}'!")
        assert False, f"Unknown input sequence mode '{mode}'!"


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
    print("Input Sequence")
    config = OmegaConf.create({
        "input_sequence": {
            "mode": "text_with_node_depths",
            "node_separator": " ",
            "do_close": False,
            "replace_newlines": False,
            "include_node_types": ["article-title", "abstract", "title", "p"]
        }
    })

    text, spans = to_input_sequence(doc, config)
    print(text)

    print("\n" * 4)

    print(spans)
    for span in spans.spans:
        print(f"'{text[span.left:span.right]}'", span.content.ntype)
