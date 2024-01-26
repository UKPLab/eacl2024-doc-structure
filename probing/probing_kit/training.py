from __future__ import annotations

import json
from pathlib import Path
from shutil import rmtree
import time
from typing import Any, Dict, List, TextIO, Tuple, Type, Union

from allennlp.data import Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.training.util import evaluate
import torch
from transformers import AutoTokenizer

from probing_kit.dataset_reader import (
    AtomicReader,
    EdgeProbingReader,
    ProbingKitReader,
    StructuralInputCreator,
    add_structural_tokens_to_tokenizer
)
from probing_kit.embedder import ProbingKitEmbedder
from probing_kit.evaluate import EvaluationForwardHook
from probing_kit.trainer import TrainerWrapper
from probing_kit.utils import probe_spans, structure_tokens, load_token_embedder_state_dict


def build_vocab(path: Path, reader: ProbingKitReader) -> Vocabulary:
    instances = []
    for dataset in ['train', 'test', 'dev']:
        instances += list(reader.read(path / dataset))
    return Vocabulary.from_instances(instances)


def get_reader(
        reader_class: Type[ProbingKitReader],
        pretrained_model_name_or_path: Union[str, Path],
        max_length: int,
        method: str = None,
        with_closing: bool = False,
        structural_input_creator: StructuralInputCreator = None,
        building_vocab: bool = False) -> ProbingKitReader:
    token_indexer = PretrainedTransformerIndexer(str(pretrained_model_name_or_path), max_length=max_length)
    if method:
        token_indexer._tokenizer.add_tokens(structure_tokens(method, with_closing))

    return reader_class(
        token_indexers={'tokens': token_indexer},
        structural_input_creator=structural_input_creator,
        building_vocab=building_vocab)

def get_data_loader(
        reader: ProbingKitReader,
        data_path: Path,
        vocab: Vocabulary,
        num_spans: int,
        batch_size: int) -> MultiProcessDataLoader:
    keys = [f'tokens_{idx + 1}' for idx in range(num_spans)] if isinstance(reader, AtomicReader) else ['tokens']
    data_loader = MultiProcessDataLoader(
        reader,
        data_path,
        batch_sampler=BucketBatchSampler(batch_size=batch_size, sorting_keys=keys),
        cuda_device=0 if torch.cuda.is_available() else None)
    data_loader.index_with(vocab)
    return data_loader


def build_data_loaders(
        reader_class: Type[ProbingKitReader],
        pretrained_model_name_or_path: Union[str, Path],
        max_length: int,
        src: Path,
        vocab: Vocabulary,
        num_spans: int,
        method: str = None,
        with_closing: bool = False,
        structural_input_creator: StructuralInputCreator = None,
        batch_size: int = 4) -> Tuple[MultiProcessDataLoader, MultiProcessDataLoader]:
    train_loader = get_data_loader(get_reader(
        reader_class,
        pretrained_model_name_or_path,
        max_length,
        method,
        with_closing,
        structural_input_creator=structural_input_creator),
        src / 'train', vocab,
        num_spans,
        batch_size)
    dev_loader = get_data_loader(get_reader(
        reader_class,
        pretrained_model_name_or_path,
        max_length,
        method,
        with_closing,
        structural_input_creator=structural_input_creator),
        src / 'dev', vocab, num_spans,
        batch_size)
    return train_loader, dev_loader


def save_json(fp: TextIO, **kwargs: Dict[Union[str, float], Any]) -> None:
    out = json.dumps(kwargs, indent=4)
    fp.write(out)


def run_training_loop(
        in_n_out: Path,
        dataset: Path,
        probes: List[str],
        pretrained_model_name_or_path: Union[str, Path],
        max_length: int = 4096,
        random_weights: bool = False,
        atomic: bool = False,
        method: str = None,
        with_closing: bool = False,
        position_embeddings_mode: str = 'vanilla',
        global_attention_mode: str = 'vanilla',
        probe_struct: bool = False,
        recover: bool = True,
        train_parameters: bool = False,
        state_dict_path: Path = None,
        lr: float = 0.1,
        batch_size: int = 4) -> None:
    assert not random_weights or not atomic

    infusion_path = f"{f'_{method}' if method is not None else ''}{'_w_closing' if with_closing else ''}{'_probe_struct' if probe_struct else ''}"
    src = dataset / f"{('atomic' if atomic else 'edge_probing')}{infusion_path if method is not None else ''}"
    spans = probe_spans()
    # The batch injector receives structural position ids, this seemed
    # easier than passing them through the allennlp abstractions
    batch_injector = {}
    # The original tokenizer is needed to tokenize the itgs in
    # exactly the same way as was done for the original dataset
    if position_embeddings_mode != 'vanilla' or global_attention_mode != 'vanilla':
        original_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        add_structural_tokens_to_tokenizer(original_tokenizer, method, with_closing)

    for probe in probes:
        if position_embeddings_mode != 'vanilla' or global_attention_mode != 'vanilla':
            # Creates structural position ids and global attention mask
            structural_input_creator = StructuralInputCreator(
                original_tokenizer,
                itg_dir=src.parent / 'intertext_graph' / probe,
                method=method,
                with_closing=with_closing,
                max_length=max_length,
                global_attention_mode=global_attention_mode,
            )
        else:
            structural_input_creator = None
        start = time.perf_counter()
        # Read dataset
        reader_class = AtomicReader if atomic else EdgeProbingReader
        vocab = build_vocab(src / probe, get_reader(
            reader_class,
            pretrained_model_name_or_path,
            max_length,
            method,
            with_closing,
            structural_input_creator=structural_input_creator,
            building_vocab=True))
        train_loader, dev_loader = build_data_loaders(
            reader_class,
            pretrained_model_name_or_path,
            max_length,
            src / probe,
            vocab,
            spans[probe],
            method,
            with_closing,
            structural_input_creator=structural_input_creator,
            batch_size=batch_size)
        # Use scalar mix to utilize all layers
        token_embedder = ProbingKitEmbedder(
            str(pretrained_model_name_or_path),
            max_length=max_length,
            train_parameters=train_parameters,
            eval_mode=not(train_parameters),
            last_layer_only=False,
            position_embeddings_mode=position_embeddings_mode,
            batch_injector=batch_injector)

        if state_dict_path is not None:
            token_embedder = load_token_embedder_state_dict(
                token_embedder,
                state_dict_path
            )
        # Train model
        serialization_dir = in_n_out / probe
        if not recover:
            rmtree(serialization_dir, ignore_errors=True)
        trainer = TrainerWrapper(
            serialization_dir,
            vocab,
            spans[probe],
            train_loader,
            dev_loader,
            token_embedder,
            random_weights,
            atomic,
            batch_injector=batch_injector,
            lr=lr)
        trainer.train()
        model = trainer.model
        # Evaluate accuracy
        test_loader = get_data_loader(get_reader(
            reader_class,
            pretrained_model_name_or_path,
            max_length,
            method,
            with_closing,
            structural_input_creator=structural_input_creator),
            src / probe / 'test',
            vocab, spans[probe],
            batch_size)
        with EvaluationForwardHook(model) as hook:
            final_metrics = evaluate(model, test_loader, cuda_device=0 if torch.cuda.is_available() else -1)
            predictions = hook.predictions
        scalar_mix = {idx: float(p.data) for idx, p in
                      enumerate(model._text_field_embedder.token_embedder_tokens._scalar_mix.scalar_parameters)}
        with (serialization_dir / 'probing.json').open('w') as f:
            save_json(
                f,
                probe={'probe_name': probe, 'duration': time.perf_counter() - start},
                metrics=final_metrics,
                scalar_mix=scalar_mix,
                predictions=predictions)
