from pathlib import Path
import random
from typing import cast, Iterator, List, Union

from intertext_graph.itgraph import Edge, Etype, IntertextDocument, Node, SpanNode

from probing_kit.dataset.pipeline import Pipeline
from probing_kit.utils import probe_spans, read_dir


class ProbingTaskPipeline(Pipeline):

    def __init__(self, *pipeline: str, out: Path, max_instances_per_label: int = None) -> None:
        """max_instances_per_label reduces the overall size of the dataset.
        Especially the LED atomic dataset can be computationally too expensive for probing.
        5 instances is a value in between the min and max number of instance in f1000rd-full and Longformer."""
        super().__init__(*pipeline, out=out)
        self._max_instances_per_label = max_instances_per_label
        self._desc = 'probing task'

    @classmethod
    def _read(cls, path: Path) -> IntertextDocument:
        """Read InterText Graphs."""
        with path.open('r') as f:
            return IntertextDocument.load_json(f)

    @classmethod
    def _annotate(
            cls,
            doc: IntertextDocument,
            label: Union[str, bool, int],
            n_1: Node,
            n_2: Node = None,
            shuffle: bool = True) -> None:
        label = {'probe': label}
        s_1 = SpanNode('span', n_1, meta={'created_by': cls.__name__}, label=label)
        doc.add_node(s_1)
        if n_2 is not None:
            # There is no label in the second span node to have a single ground truth value
            s_2 = SpanNode('span', n_2, meta={'created_by': cls.__name__})
            doc.add_node(s_2)
            # Shuffle nodes so no clues can be inferred from their order
            span_nodes = [s_1, s_2]
            if shuffle:
                random.shuffle(span_nodes)
                # If the source node does not contain a label swap labels
                if not span_nodes[0].label:
                    span_nodes[0].label, span_nodes[1].label = span_nodes[1].label, span_nodes[0].label
            edge = Edge(*span_nodes, etype=Etype.LINK, meta={'created_by': cls.__name__})
            doc.add_edge(edge)

    @classmethod
    def _add_node_type_probe(cls, doc: IntertextDocument) -> IntertextDocument:
        for node in doc.unroll_graph():
            if node.ntype in ['title', 'p']:
                # Differentiate between depth of sections
                if node.ntype == 'title':
                    label = 'section'
                    depth = doc.tree_distance(doc.root, node, Etype.PARENT)
                    if depth > 1:
                        # Bundling all subsubsubsections due to sparsity of data
                        # Differentiating additional levels is handled by the tree depth probe
                        label = 'subsection'
                else:
                    label = 'paragraph'
                cls._annotate(doc, label, node)
        return doc

    @classmethod
    def _add_structural_probe(cls, doc: IntertextDocument, etype: Etype = Etype.PARENT) -> IntertextDocument:
        nodes = doc.unroll_graph()
        for idx, n_1 in enumerate(nodes):
            for n_2 in nodes[idx + 1:]:
                tree_distance = doc.tree_distance(n_1, n_2, etype)
                cls._annotate(doc, tree_distance, n_1, n_2)
        return doc

    @classmethod
    def _add_tree_depth_probe(cls, doc: IntertextDocument) -> IntertextDocument:
        root = doc.root
        for node in doc.unroll_graph()[1:]:
            tree_distance = doc.tree_distance(root, node, Etype.PARENT)
            cls._annotate(doc, tree_distance, root, node)
        return doc

    @classmethod
    def _add_sibling_probe(cls, doc: IntertextDocument) -> IntertextDocument:
        root = doc.root
        nodes = doc.unroll_graph()
        for idx, n_1 in enumerate(nodes):
            for n_2 in nodes[idx + 1:]:
                if n_1 == root or n_2 == root:
                    # There is a single root therefore n_1 and n_2 cannot be siblings
                    label = False
                else:
                    n_1_parent = n_1.get_edges(Etype.PARENT, outgoing=False)[0].src_node
                    n_2_parent = n_2.get_edges(Etype.PARENT, outgoing=False)[0].src_node
                    label = n_1_parent == n_2_parent
                # Shuffle to get bi-directional samples
                cls._annotate(doc, label, n_1, n_2)
        return doc

    @classmethod
    def _add_ancestor_probe(cls, doc: IntertextDocument) -> IntertextDocument:
        nodes = doc.unroll_graph()
        for n_1 in nodes:
            # Loop through all permutations to get negative directed samples
            for n_2 in nodes:
                if n_1 == n_2:
                    continue
                label = False
                for ancestor in doc.breadcrumbs(n_2, Etype.PARENT):
                    if ancestor == n_1:
                        label = True
                        break
                # Fix position for asymmetrical probes
                cls._annotate(doc, label, n_1, n_2, shuffle=False)
        return doc

    @classmethod
    def _add_parent_predecessor_probe(cls, doc: IntertextDocument, etype: Etype = Etype.PARENT) -> IntertextDocument:
        root = doc.root
        nodes = doc.unroll_graph()
        for n_1 in nodes:
            # Loop through all permutations to get negative directed samples
            for n_2 in nodes:
                # Root has no predecessors
                if n_1 == n_2 or n_2 == root:
                    continue
                n_2_predecessor = n_2.get_edges(etype, outgoing=False)[0].src_node
                # Fix position for asymmetrical probes
                cls._annotate(doc, n_1 == n_2_predecessor, n_1, n_2, shuffle=False)
        return doc

    @classmethod
    def _add_position_probe(cls, doc: IntertextDocument) -> IntertextDocument:
        nodes = doc.unroll_graph()
        for node in nodes:
            descendants = {nodes.index(e.tgt_node): e.tgt_node for e in node.get_edges(Etype.PARENT, incoming=False)}
            if len(descendants) > 1:
                begin, *inside, end = sorted(list(descendants.keys()))
                cls._annotate(doc, 'begin', descendants[begin])
                for n in inside:
                    cls._annotate(doc, 'inside', descendants[n])
                cls._annotate(doc, 'end', descendants[end])
        return doc

    def _select_overhead(self, lst: List[Union[Node, Edge]], num_nodes: int) -> Iterator[Union[SpanNode, Edge]]:
        labels = {}
        for value in lst:
            if not value.meta:
                print()
            if value.meta and value.meta['created_by'] == type(self).__name__ and (isinstance(value, SpanNode) or (
                    isinstance(value.src_node, SpanNode) and isinstance(value.tgt_node, SpanNode))):
                node = cast(SpanNode, value.src_node if isinstance(value, Edge) else value)
                # There are span nodes and edges without probing labels
                if 'probe' in node.label:
                    label = node.label['probe']
                    if label not in labels:
                        labels[label] = []
                    labels[label].append(value)
        if len(labels) > 0:
            # Shuffle and yield excess values
            # Upper bound per label is the smallest label bucket or max_instances_per_label
            # Upper bound of total labels is the number of all nodes
            k = min(min(map(len, labels.values())), num_nodes // len(labels))
            if self._max_instances_per_label is not None:
                k = min(k, self._max_instances_per_label)
            for bucket in labels.values():
                random.shuffle(bucket)
                for value in bucket[k:]:
                    yield value

    def _balance(self, doc: IntertextDocument) -> IntertextDocument:
        random.seed(0)
        num_nodes = len(doc.unroll_graph())
        for edge in self._select_overhead(doc.edges, num_nodes):
            # Remove src and tgt node which will automatically remove the edge
            # When removing just the edge their respective span nodes would remain
            src_node = edge.src_node
            tgt_node = edge.tgt_node
            doc.remove_node(src_node)
            doc.remove_node(tgt_node)
        for node in self._select_overhead(doc.nodes, num_nodes):
            doc.remove_node(node)
        return doc

    def _save(self, doc: IntertextDocument) -> None:
        with (self.out / f'{doc.meta["doc_id"]}_v{doc.meta["version"]}.json').open('w') as f:
            doc.save_json(f)

    @classmethod
    def run_default_config(
            cls,
            dataset: Path,
            recover: bool = True) -> None:
        files = read_dir(dataset / 'preprocessed', '.json')
        for probe_name in probe_spans().keys():
            pipeline = cls(
                'read',
                f'add_{probe_name}_probe',
                'balance',
                out=dataset / 'intertext_graph' / probe_name)
            if not recover or not pipeline.out.exists():
                pipeline(files)
