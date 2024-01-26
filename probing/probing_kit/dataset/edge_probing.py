import json
from pathlib import Path
from typing import cast, List, Tuple, Union

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.token_class import Token
from intertext_graph.itgraph import Etype, IntertextDocument, SpanNode

from probing_kit.dataset.probing_mixin import ProbingMixin
from probing_kit.dataset.pipeline import Pipeline
from probing_kit.tokenizer import DisableSpecialTokens
from probing_kit.utils import probe_spans, read_dir, structure_tokens, train_test_dev_split


class EdgeProbingPipeline(Pipeline):

    def __init__(
            self,
            *pipeline: str,
            out: Path,
            tokenizer: PretrainedTransformerTokenizer,
            concat_separator: str = ' ',
            probe_structure: bool = False) -> None:
        super().__init__(*pipeline, out=out)
        self._tokenizer = tokenizer
        # Pre-processing of Longformer data followed that of RoBERTa where they concatenated lines
        self._concat_separator = concat_separator
        self._probe_structure = probe_structure
        self._desc = 'edge probing'

    @classmethod
    def _read(cls, path: Path) -> IntertextDocument:
        """Read InterText Graphs."""
        with path.open('r') as f:
            return IntertextDocument.load_json(f)

    def _find_span_indices(
            self,
            outer: List[Token],
            inner: List[Token],
            offset: int = 0,
            expansions: int = 2) -> Tuple[int, int]:
        inner, outer = [str(token) for token in inner], [str(token) for token in outer]
        span = self._tokenizer.tokenizer.convert_tokens_to_string(inner)
        for start in range(offset, len(outer)):
            # Do substring matching due to potential preceding special tokens, e.g. whitespace
            if inner[0] in ''.join(outer[start:start + expansions + 1]):
                # Allow additional expansions before a substring match has to occur
                repeat = 0
                # Expand candidate
                for end in range(start + 1, len(outer) + 1):
                    candidate = self._tokenizer.tokenizer.convert_tokens_to_string(outer[start:end])
                    # Allow the first char of a token
                    if candidate[1:] not in span:
                        # Break if the candidate does not match within two expansions
                        if repeat <= expansions:
                            repeat += 1
                            continue
                        else:
                            break
                    elif candidate == span or candidate[1:] == span:
                        # Adjust the span to match a leading whitespace that is not part of the structure token
                        if self._probe_structure and candidate[1:] == span:
                            start += 1
                        # Return span match
                        return start, end
                    repeat = 0
        assert False

    def _save(self, doc: IntertextDocument) -> None:
        tokens = self._tokenizer.tokenize(self._concat_separator.join([str(node) for node in doc.unroll_graph()]))
        targets = {}
        info = {'source': 'F1000Research'} | doc.meta
        start, end = 0, 0
        for idx, node in enumerate(doc.unroll_graph()):
            # Find span indices of span without special tokens
            with DisableSpecialTokens(self._tokenizer) as t:
                tokenized_node = t.tokenize(str(node))
            prev = end
            start, end = self._find_span_indices(tokens, tokenized_node, end)
            # Each node should succeed its predecessor without gaps
            # Allowing gaps of 1 for special token and additional ones at the first position
            assert start == prev or start == prev + 1 or start == prev + 2 or prev == 0
            # Extract edge probes from linked SpanNodes
            for link in node.get_edges(etype=Etype.LINK, incoming=False):
                # Check for span ntype to skip reference links
                if link.tgt_node.ntype != 'span':
                    continue
                span_node = cast(SpanNode, link.tgt_node)
                # Find span indices of span without special tokens
                with DisableSpecialTokens(self._tokenizer) as t:
                    tokenized_span = t.tokenize(str(span_node))
                # Find the span in the full document based on the current node start offset
                span = self._find_span_indices(tokens, tokenized_span, start)
                # Double spans are linked with a link edge, i.e. span1 -> span2
                # Get a unique ix for either a single (current node) or double span (src_node) probe
                # Retrieve ix and label from either the node or edge depending on a single or double span
                span_ix = span_node.ix
                label = span_node.label['probe'] if 'probe' in span_node.label else None
                for edge in span_node.get_edges(etype=Etype.LINK):
                    if isinstance(edge.src_node, SpanNode) and isinstance(edge.tgt_node, SpanNode):
                        span_ix = edge.src_node.ix
                        label = cast(SpanNode, edge.src_node).label['probe']
                        break
                # Skip if there is no associated label
                if label is None:
                    continue
                # Determine whether the current node is span1 or span2
                # Some labels depend on the order of spans
                key = 'span1' if span_node.ix == span_ix else 'span2'
                if span_ix not in targets:
                    targets[span_ix] = {key: span, 'label': label}
                else:
                    assert label == targets[span_ix]['label']
                    targets[span_ix][key] = span
        # Offset found in the last node should always equal the token length
        # Minus gaps of 1 for special token and additional ones at the last position
        assert end == len(tokens) - 1 or end == len(tokens) - 2
        with (self.out / f'{doc.meta["doc_id"]}_v{doc.meta["version"]}.json').open('w') as f:
            json.dump({
                'tokens': [{'text': token.text, 'text_id': token.text_id, 'type_id': token.type_id, 'idx': token.idx, 'idx_end': token.idx_end} for token in tokens],
                'targets': list(targets.values()),
                'info': info}, f, indent=4)

    @classmethod
    def run_default_config(
            cls,
            dataset: Path,
            pretrained_model_name_or_path: Union[str, Path],
            method: str = None,
            with_closing: bool = False,
            probe_structure: bool = False,
            recover: bool = True) -> None:
        # Do NOT use max_length as it would trim the doc making the token_filter() method useless
        tokenizer = PretrainedTransformerTokenizer(str(pretrained_model_name_or_path))
        with_structure_infusion = method is not None
        if with_structure_infusion:
            tokenizer.tokenizer.add_tokens(structure_tokens(method, with_closing))
        infusion_path = f"{f'_{method}' if method is not None else ''}{'_w_closing' if with_closing else ''}{'_probe_struct' if probe_structure else ''}"
        for probe_name in probe_spans().keys():
            files = read_dir(dataset / (infusion_path[1:] if with_structure_infusion else 'intertext_graph') / probe_name, '.json')
            for key, value in train_test_dev_split(files).items():
                pipeline = EdgeProbingPipeline(
                    'read',
                    out=dataset / f'edge_probing{infusion_path}' / probe_name / key,
                    tokenizer=tokenizer,
                    probe_structure=probe_structure)
                if not recover or not pipeline.out.exists():
                    pipeline(value)
                atomic = AtomicPipeline(
                    'read',
                    out=dataset / f'atomic{infusion_path}' / probe_name / key,
                    tokenizer=tokenizer)
                if not recover or not atomic.out.exists():
                    atomic(value)


class AtomicPipeline(EdgeProbingPipeline):

    def __init__(
            self,
            *pipeline: str,
            out: Path,
            tokenizer: PretrainedTransformerTokenizer) -> None:
        super().__init__(*pipeline, out=out, tokenizer=tokenizer)
        self._desc = 'atomic'

    def _save(self, doc: IntertextDocument) -> None:
        doc_spans = {'spans': [], 'info': {'source': 'F1000Research'} | doc.meta}
        for node in doc.unroll_graph():
            for link in node.get_edges(etype=Etype.LINK, incoming=False):
                # Check for span ntype to skip reference links
                if link.tgt_node.ntype != 'span':
                    continue
                span_node = cast(SpanNode, link.tgt_node)
                label = span_node.label['probe'] if 'probe' in span_node.label else None
                span_nodes = [span_node]
                for edge in span_node.get_edges(etype=Etype.LINK):
                    if isinstance(edge.src_node, SpanNode) and isinstance(edge.tgt_node, SpanNode):
                        if edge.src_node in span_nodes:
                            span_nodes.append(edge.tgt_node)
                        else:
                            span_nodes = [edge.src_node] + span_nodes
                        label = cast(SpanNode, edge.src_node).label['probe']
                tokens = [self._tokenizer.tokenize(str(node)) for node in span_nodes]
                targets = {'label': label}
                start = 0
                for idx, n in enumerate(span_nodes):
                    # Find span indices of span without special tokens
                    with DisableSpecialTokens(self._tokenizer) as t:
                        tokenized_span = t.tokenize(str(n))
                    # Find the span in the full document based on the current node start offset
                    start, end = self._find_span_indices(tokens[idx], tokenized_span, start)
                    targets[f'span{idx + 1}'] = [start, end]
                doc_spans['spans'].append({
                    'tokens': [[{'text': token.text, 'text_id': token.text_id, 'type_id': token.type_id, 'idx': token.idx, 'idx_end': token.idx_end} for token in span] for span in tokens],
                    'targets': targets,
                })
                # Removing span nodes prevents extracting the same data twice for double spans
                # Propagates to attached span edges
                for n in span_nodes:
                    doc.remove_node(n)
        with (self.out / f'{doc.meta["doc_id"]}_v{doc.meta["version"]}.json').open('w') as f:
            json.dump(doc_spans, f, indent=4)


class EdgeProbingInfusionPipeline(EdgeProbingPipeline, ProbingMixin):

    def __init__(
            self,
            *pipeline: str,
            out: Path,
            tokenizer: PretrainedTransformerTokenizer,
            max_length: int) -> None:
        # Structure tags are present as separators making additional separation chars redundant
        super().__init__(*pipeline, out=out, tokenizer=tokenizer, concat_separator='')
        self._max_length = max_length
        self._desc = 'edge probing infusion'
