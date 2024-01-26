from multiprocessing import Manager
from pathlib import Path
from typing import Union

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from intertext_graph.itgraph import Etype, IntertextDocument, Node

from probing_kit.dataset.probing_mixin import ProbingMixin
from probing_kit.dataset.pipeline import Pipeline
from probing_kit.utils import probe_spans, read_dir, structure_tokens


class InfusionPipeline(Pipeline, ProbingMixin):

    def __init__(
            self,
            *pipeline: str,
            out: Path,
            tokenizer: PretrainedTransformerTokenizer,
            max_length: int,
            method: str,
            with_closing: bool = False,
            probe_structure: bool = False) -> None:
        super().__init__(*pipeline, out=out)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._method = method
        self._with_closing = with_closing
        self._probe_structure = probe_structure
        self._desc = 'infusion'
        manager = Manager()
        self._save_request_lock = manager.Lock()

    @classmethod
    def _read(cls, path: Path) -> IntertextDocument:
        """Read InterText Graphs."""
        with path.open('r') as f:
            return IntertextDocument.load_json(f)

    def _infuse_structure(self, doc: IntertextDocument) -> IntertextDocument:
        return getattr(self, f'_add_{self._method}')(doc)

    def _update_span_nodes(self, node: Node, offset: int = 0) -> None:
        # Update SpanNode spans for new content
        for edge in node.get_edges(Etype.LINK, incoming=False):
            if not self._probe_structure:
                edge.tgt_node.start = offset
            edge.tgt_node.end = len(str(node))

    def _add_node_boundaries(self, doc: IntertextDocument) -> IntertextDocument:
        tag = '<node>'
        for node in doc.unroll_graph():
            node.content = f'{tag} {node.content}'
            self._update_span_nodes(node, len(tag) + 1)
        return doc

    def _add_node_types(self, doc: IntertextDocument) -> IntertextDocument:
        stack = []
        last = None
        # Use offset to set spans for first and last text token, ignoring structure tokens
        offset = 0
        for node in doc.unroll_graph():
            node.content = f'<{node.ntype}> {node.content}'
            offset += len(node.ntype) + 3
            if self._with_closing:
                if stack:
                    parent = node.get_edges(Etype.PARENT, outgoing=False)[0].src_node
                    tmp = ''
                    for idx in range(len(stack) - stack.index(parent) - 1):
                        tmp += f'</{stack.pop().ntype}> '
                    node.content = f'{tmp}{node.content}'
                    offset += len(tmp)
                stack.append(node)
            self._update_span_nodes(node, offset)
            offset = 0
            last = node
        if self._with_closing:
            for _ in range(len(stack)):
                last.content += f' </{stack.pop().ntype}>'
            # Do not update span node indices
        return doc

    def _add_node_depth(self, doc: IntertextDocument) -> IntertextDocument:
        for node in doc.unroll_graph():
            depth = len(list(doc.breadcrumbs(node, Etype.PARENT)))
            node.content = f'<node-{depth}> {node.content}'
            self._update_span_nodes(node, len(f'<node-{depth}>') + 1)
        return doc

    def _save(self, doc: IntertextDocument) -> None:
        with (self.out / f'{doc.meta["doc_id"]}_v{doc.meta["version"]}.json').open('w') as f:
            doc.save_json(f)

    @classmethod
    def run_default_config(
            cls,
            dataset: Path,
            pretrained_model_name_or_path: Union[str, Path],
            max_length: int = 4096,
            method: str = 'node_boundaries',
            with_closing: bool = False,
            probe_structure: bool = False,
            recover: bool = True) -> None:
        # When closing tags are set it is unclear which tags to probe as closing tags can be in a different node
        assert not with_closing or not probe_structure
        tokenizer = PretrainedTransformerTokenizer(str(pretrained_model_name_or_path))
        # Do NOT use max_length as it would trim the doc making the token_filter() method useless
        tokenizer.tokenizer.add_tokens(structure_tokens(method, with_closing))
        for probe_name in probe_spans().keys():
            files = read_dir(dataset / 'intertext_graph' / probe_name, '.json')
            pipeline = cls(
                'read',
                'infuse_structure',
                'token_filter',
                out=dataset / f"{method}{'_w_closing' if with_closing else ''}{'_probe_struct' if probe_structure else ''}" / probe_name,
                tokenizer=tokenizer,
                method=method,
                with_closing=with_closing,
                probe_structure=probe_structure,
                max_length=max_length)
            if not recover or not pipeline.out.exists():
                pipeline(files)
