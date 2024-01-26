from __future__ import annotations

import collections
from abc import ABC
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Union

import torch
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, ListField, SpanField, TextField, ArrayField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from intertext_graph.itgraph import IntertextDocument, Etype
from transformers import PreTrainedTokenizerFast, BatchEncoding

from probing_kit.utils import read_dir
from probing_kit.tokenizer import PreTokenizer

try:
    from infusion.structformer.input_sequence import to_input_sequence, get_structural_tokens
    from infusion.structformer.global_attention import make_global_attention_mask
    from infusion.structformer.tokenized_structure import make_tokenized_structure
    infusion_available = True
except (ImportError, ModuleNotFoundError):
    infusion_available = False


class ProbingKitReader(DatasetReader, ABC):

    def __init__(
        self,
        tokenizer: PreTokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        structural_input_creator: StructuralInputCreator = None,
        building_vocab: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or PreTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.structural_input_creator = structural_input_creator
        # This determines whether structural inputs are created, as this
        # is not needed when building the vocab
        self.building_vocab = building_vocab

    def _read(self, path: Union[Path, str]) -> Iterable[Instance]:
        random.seed(0)
        files = read_dir(path, '.json')
        random.shuffle(files)
        for file in files:
            with file.open('r') as f:
                data = json.load(f)
                yield from self._read_probing_data(data, filename=file.name)

    def _read_probing_data(self, data: Dict[str, Any], filename: str) -> Iterable[Instance]:
        pass


class EdgeProbingReader(ProbingKitReader):

    def text_to_instance(
            self,
            tokens: List[Dict[str, Union[str, int]]],
            labels: List[str],
            spans_1: List[List[int]],
            spans_2: List[List[int]] = None) -> Instance:
        tokens = self.tokenizer.tokenize(tokens)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        labels_field = ListField([LabelField(label) for label in labels])
        # Subtract from end index since SpanField uses inclusive indexing
        spans_1_field = ListField([SpanField(span[0], span[1] - 1, text_field) for span in spans_1])
        fields = {
            'tokens': text_field,
            'labels': labels_field,
            'spans_1': spans_1_field}
        if spans_2 is not None:
            # Subtract from end index since SpanField uses inclusive indexing
            fields['spans_2'] = ListField([SpanField(span[0], span[1] - 1, text_field) for span in spans_2])
        return Instance(fields)

    def _read_probing_data(self, data: Dict[str, Any], filename: str = None) -> Iterable[Instance]:
        tokens = data['tokens']
        labels = [str(target['label']) for target in data['targets']]
        spans_1 = [target['span1'] for target in data['targets']]
        if 'span2' in data['targets'][0]:
            spans_2 = [target['span2'] for target in data['targets']]
        else:
            spans_2 = None
        instance = self.text_to_instance(tokens, labels, spans_1, spans_2)
        if self.structural_input_creator is not None and not self.building_vocab:
            structural_input = (
                self.structural_input_creator.get_structural_input(
                filename, instance.fields['tokens']
            ))
            instance.add_field('node_types_ids', ArrayField(structural_input['node_types_ids']))
            instance.add_field('node_depths_ids', ArrayField(structural_input['node_depths_ids']))
            instance.add_field('global_attention_mask', ArrayField(structural_input['global_attention_mask']))

        yield instance


class AtomicReader(ProbingKitReader):

    def text_to_instance(
            self,
            tokens_1: List[Dict[str, Union[str, int]]],
            label: str,
            span_1: List[int],
            tokens_2: List[Dict[str, Union[str, int]]] = None,
            span_2: List[int] = None) -> Instance:
        tokens_1 = self.tokenizer.tokenize(tokens_1)
        if self.max_tokens:
            tokens_1 = tokens_1[:self.max_tokens]
        text_field_1 = TextField(tokens_1, self.token_indexers)
        label_field = LabelField(label)
        # Subtract from end index since SpanField uses inclusive indexing
        span_1_field = SpanField(span_1[0], span_1[1] - 1, text_field_1)
        fields = {
            'tokens_1': text_field_1,
            'label': label_field,
            'span_1': span_1_field}
        if tokens_2 is not None and span_2 is not None:
            tokens_2 = self.tokenizer.tokenize(tokens_2)
            if self.max_tokens:
                tokens_2 = tokens_2[:self.max_tokens]
            fields['tokens_2'] = TextField(tokens_2, self.token_indexers)
            # Subtract from end index since SpanField uses inclusive indexing
            fields['span_2'] = SpanField(span_2[0], span_2[1] - 1, fields['tokens_2'])
        return Instance(fields)

    def _read_probing_data(self, data: Dict[str, List[Dict[str, Any]]], filename: str = None) -> Iterable[Instance]:
        for span in data['spans']:
            tokens_1 = span['tokens'][0]
            label = str(span['targets']['label'])
            span_1 = span['targets']['span1']
            if len(span['tokens']) > 1:
                tokens_2 = span['tokens'][1]
                span_2 = span['targets']['span2']
            else:
                tokens_2 = None
                span_2 = None

            instance = self.text_to_instance(tokens_1, label, span_1, tokens_2, span_2)

            if self.structural_input_creator is not None and not self.building_vocab:
                structural_input = (
                    self.structural_input_creator.get_atomic_structural_input(
                        filename,
                        instance.fields['tokens_1'],
                        tokens_2=instance.fields['tokens_2'] if 'tokens_2' in instance.fields else None,
                    )
                )
                instance.add_field('node_types_ids_1', ArrayField(structural_input['node_types_ids_1']))
                instance.add_field('node_depths_ids_1', ArrayField(structural_input['node_depths_ids_1']))
                instance.add_field('global_attention_mask_1', ArrayField(structural_input['global_attention_mask_1']))
                if tokens_2 is not None:
                    instance.add_field('node_types_ids_2', ArrayField(structural_input['node_types_ids_2']))
                    instance.add_field('node_depths_ids_2', ArrayField(structural_input['node_depths_ids_2']))
                    instance.add_field('global_attention_mask_2', ArrayField(structural_input['global_attention_mask_2']))
            yield instance


############################################################################################################
# Added by
############################################################################################################

class StructuralInputCreator:
    """Creates structural position ids and global attention mask with the help of the relpos_graph
    repository. Currently implemented structural position ids are:
    - node type ids
    - node depth ids
    """
    def __init__(
            self,
            original_tokenizer: PreTrainedTokenizerFast,
            itg_dir: Path = None,
            method: str = None,
            with_closing: bool = False,
            max_length: int = None,
            input_sequence_include_node_types: List[str] = None,
            input_sequence_use_bos_eos_token: bool = False,
            input_sequence_bos_token: str = '<s>',
            input_sequence_eos_token: str = '</s>',
            global_attention_mode: str = 'vanilla'
    ):

        self.itg_dir = itg_dir
        self.original_tokenizer = original_tokenizer
        self.max_length = max_length

        if input_sequence_include_node_types is None:
            input_sequence_include_node_types = [
                'article-title',
                'abstract',
                'title',
                'p'
            ]

        self.node_types_map = {
            node_type: i + 1
            for i, node_type in enumerate(input_sequence_include_node_types)
        }

        self.config = _make_relpos_graph_config(
            method=method,
            with_closing=with_closing,
            input_sequence_include_node_types=input_sequence_include_node_types,
            input_sequence_use_bos_eos_token=input_sequence_use_bos_eos_token,
            input_sequence_bos_token=input_sequence_bos_token,
            input_sequence_eos_token=input_sequence_eos_token,
            global_attention_mode=global_attention_mode
        )
        self.global_attention_mode = global_attention_mode
        self.method = method
        self.with_closing = with_closing

        self.structural_tokens = get_structural_tokens(self.config)



    def get_structural_input(
            self,
            filename: IntertextDocument,
            tokens: TextField,
            atomic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Get the structural position ids for a given ITG file.
        The structural information is missing in the edge probing instance.
        Therefore, we load the corresponding ITG file, tokenize it again,
        and use the resulting data structures to determine the structural
        position ids of each token.
        The tokens from the probing instance are used to determine the length of the
        resulting position ids tensors and to ensure the correct tokenization
        of the ITG file.
        """
        # Load ITG file
        if filename is not None:
            itg_path = self.itg_dir / filename
            with itg_path.open('r') as f:
                itg = IntertextDocument.load_json(f)
        else:
            return {
                'node_types_ids': None,
                'node_depths_ids': None,
                'global_attention_mask': None
            }

        # Create dummy instance object to be compatible with relpos_graph code
        instance_class = collections.namedtuple('Instance', 'document')
        instance = instance_class(itg)
        instances = [instance]

        # Copied from structformer.scaffold_tasks.collate_s2orcitgsubset_instances
        # Linearize the document and create a list of node spans
        text_and_spans = [to_input_sequence(instance.document, self.config) for instance in instances]
        texts = [text for text, _ in text_and_spans]
        node_spans = [spans for _, spans in text_and_spans]
        offsets = [0 for _ in text_and_spans]

        max_length = tokens.sequence_length()

        # Tokenize ITG file
        inputs = self.original_tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation="longest_first",
            max_length=max_length,
            add_special_tokens=True
        )
        if not atomic:
            if len(inputs[0].tokens) != len(tokens.tokens):
                raise ValueError(
                    'Tokenization of ITG file does not match tokenization '
                    'of probing instance.'
                )
            for i in range(len(inputs[0].tokens)):
                if (
                        inputs[0].tokens[i] != str(tokens.tokens[i]) \
                        and not str(tokens.tokens[i]) == '<unk>' \
                ):
                    raise ValueError(
                        'Tokenization of ITG file does not match tokenization of probing instance.'
                    )

        # Make global attention mask
        make_global_attention_mask(
            instances=instances,
            node_spans=node_spans,
            inputs=inputs,
            offsets=offsets,
            tokenizer=self.original_tokenizer,
            config=self.config
        )

        # Get structural position ids
        tokenized_structure = make_tokenized_structure(
            instances=instances,
            node_spans=node_spans,
            inputs=inputs,
            offsets=offsets,
            config=self.config
        )
        node_types_ids = tokenized_structure['node_types_labels']
        node_depths_ids = tokenized_structure['node_depths_labels']

        # Truncate to max length
        if self.max_length is not None:
            node_types_ids = node_types_ids[:, :self.max_length]
            node_depths_ids = node_depths_ids[:, :self.max_length]

        return {
            'node_types_ids': node_types_ids[0],
            'node_depths_ids': node_depths_ids[0],
            'global_attention_mask': inputs['global_attention_mask'][0]
        }

    def get_atomic_structural_input(
            self,
            filename: str,
            tokens_1: TextField,
            tokens_2: TextField = None,
    ) -> Dict[str, torch.Tensor]:
        """Get the structural position ids for a given ITG file.
        """
        if filename is not None:
            itg_path = self.itg_dir / filename
            with itg_path.open('r') as f:
                itg = IntertextDocument.load_json(f)
        else:
            return {
                'node_types_ids_1': None,
                'node_depths_ids_1': None,
                'global_attention_mask_1': None,
                'node_types_ids_2': None,
                'node_depths_ids_2': None,
                'global_attention_mask_2': None,
            }

        node_depths_ids_1, node_types_ids_1, global_attention_mask_1 = self.get_structural_input_for_node(
            tokens_1,
            itg
        )
        if tokens_2 is not None:
            node_depths_ids_2, node_types_ids_2, global_attention_mask_2 = self.get_structural_input_for_node(
                tokens_2,
                itg
            )
        else:
            node_types_ids_2 = None
            node_depths_ids_2 = None
            global_attention_mask_2 = None

        return {
            'node_types_ids_1': node_types_ids_1,
            'node_depths_ids_1': node_depths_ids_1,
            'global_attention_mask_1': global_attention_mask_1,
            'node_types_ids_2': node_types_ids_2,
            'node_depths_ids_2': node_depths_ids_2,
            'global_attention_mask_2': global_attention_mask_2,
        }

    def get_structural_input_for_node(
            self,
            tokens: TextField,
            itg: IntertextDocument,
    ):
        # Remove structural tokens from beginning and end
        text_tokens = [
            str(t) for t in tokens.tokens[1:-1]
            if not str(t) in self.structural_tokens
        ]
        text = self.original_tokenizer.convert_tokens_to_string(
            text_tokens
        ).strip()

        node_found = False
        for i, node in enumerate(itg.nodes):
            if node.content == text:
                node_found = True
                break

        if not node_found:
            raise RuntimeError(f'Could not find node with text {text}')

        node_type = node.ntype
        depth = len(list(itg.breadcrumbs(node, Etype.PARENT)))

        node_depths_ids = torch.zeros(len(tokens.tokens), dtype=torch.long)
        node_depths_ids[1:-1] = depth
        node_types_ids = torch.zeros(len(tokens.tokens), dtype=torch.long)
        node_types_ids[1:-1] = self.node_types_map[node_type]

        global_attention_mask = torch.zeros(len(tokens.tokens), dtype=torch.long)
        global_attention_mask[0] = 1
        if self.global_attention_mode == 'global_attention_on_first_tokens':
            global_attention_mask[1] = 1
        elif self.global_attention_mode == 'global_attention_on_structural_tokens':
            global_attention_mask[1] = 1
            if self.with_closing:
                global_attention_mask[-2] = 1

        return node_depths_ids, node_types_ids, global_attention_mask

def add_structural_tokens_to_tokenizer(
        tokenizer: PreTrainedTokenizerFast,
        method: str = None,
        with_closing: bool = False,
        input_sequence_include_node_types: List[str] = None,
):
    """Add structural tokens to the tokenizer vocabulary."""


    config = _make_relpos_graph_config(
        method=method,
        with_closing=with_closing,
        input_sequence_include_node_types=input_sequence_include_node_types
    )

    structural_tokens = get_structural_tokens(config)

    tokenizer.add_tokens(structural_tokens)

    return


def _make_relpos_graph_config(
        method: str = None,
        with_closing: bool = False,
        input_sequence_include_node_types: List[str] = None,
        input_sequence_use_bos_eos_token: bool = False,
        input_sequence_bos_token: str = '<s>',
        input_sequence_eos_token: str = '</s>',
        global_attention_mode: str = 'vanilla'
) -> Dict[str, Any]:
    """Create a config object for the relpos_graph code."""
    if input_sequence_include_node_types is None:
        input_sequence_include_node_types = [
            'article-title',
            'abstract',
            'title',
            'p'
        ]
    if sorted(input_sequence_include_node_types) == sorted(['article-title', 'abstract', 'title', 'p']):
        input_sequence_use_core_node_types_only = True
    else:
        input_sequence_use_core_node_types_only = False

    # Determine input sequence mode
    if method is None:
        input_sequence_mode = 'vanilla'
    elif method == 'node_boundaries':
        input_sequence_mode = 'text_with_node_boundaries'
    elif method == 'node_depth':
        input_sequence_mode = 'text_with_node_depths'
    elif method == 'node_types':
        input_sequence_mode = 'text_with_node_types'
    else:
        raise ValueError(f'Unknown method: {method}')
    input_sequence_do_close = with_closing

    # Dummy config object to be compatible with relpos_graph code
    config = {
        'max_depth': 20,
        'node_types': input_sequence_include_node_types,
        'input_sequence': {
            'mode': input_sequence_mode,
            'include_node_types': input_sequence_include_node_types,
            'replace_newlines': False,
            'node_separator': ' ',
            'do_close': input_sequence_do_close,
            'use_bos_eos_token': input_sequence_use_bos_eos_token,
            'bos_token': input_sequence_bos_token,
            'eos_token': input_sequence_eos_token,
            'use_core_node_types_only': input_sequence_use_core_node_types_only,
        },
        'attention': {
            'mode': global_attention_mode
        },
        'model': {
            'task_name': 'Scaffold Task For Pretraining'
        },
        'position_embeddings': {
            'mode': ''
        }
    }

    return config