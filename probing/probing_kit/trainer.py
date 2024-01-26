from itertools import chain
from pathlib import Path
from typing import Any, Dict

from allennlp.data import Vocabulary
from allennlp.data.data_loaders import DataLoader, MultiProcessDataLoader
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
import numpy as np
import torch

from probing_kit.classifier import AtomicModel, ProbingModel
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.training.optimizers import AdamOptimizer


class TrainerWrapper:

    def __init__(
            self,
            serialization_dir: Path,
            vocab: Vocabulary,
            num_spans: int,
            train_loader: MultiProcessDataLoader,
            dev_loader: MultiProcessDataLoader,
            token_embedder: PretrainedTransformerEmbedder,
            random_weights: bool = False,
            atomic: bool = False,
            batch_injector: Dict = None,
            lr: float = 0.1) -> None:
        self._serialization_dir = serialization_dir  # This is the parent of the actual serialization dir
        self._vocab = vocab
        self._num_spans = num_spans
        self._train_loader = train_loader
        self._dev_loader = dev_loader
        self._token_embedder = token_embedder
        self._random_weights = random_weights
        self._atomic = atomic
        self._trainer = None
        self.model = None
        self.batch_injector = batch_injector
        self._lr = lr

    def _build_model(self) -> ProbingModel:
        assert not (self._random_weights and self._atomic)
        if self._random_weights:
            self._token_embedder.transformer_model.apply(self._init_weights)
        # Extract and concatenate the first and last token of each span
        span_extractor = EndpointSpanExtractor(self._token_embedder.get_output_dim(), 'x,y')
        model_class = AtomicModel if self._atomic else ProbingModel
        model = model_class(self._vocab, self._token_embedder, span_extractor, self._num_spans, batch_injector=self.batch_injector)
        if torch.cuda.is_available():
            model.cuda()
        return model

    def _build_trainer(self) -> GradientDescentTrainer:
        checkpointer = Checkpointer(self._serialization_dir, keep_most_recent_by_count=2)
        # Most hyperparameters based on Hewitt and Manning, 2019
        parameters = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = AdamOptimizer(
            model_parameters=parameters,
            lr=self._lr,
            betas=(0.9, 0.999),
            eps=1e-08)
        num_gradient_accumulation_steps = max(1, 64 // self._train_loader.batch_sampler.get_batch_size())
        return GradientDescentTrainer(
            model=self.model,
            serialization_dir=self._serialization_dir,
            checkpointer=checkpointer,
            data_loader=self._train_loader,
            validation_data_loader=self._dev_loader,
            num_epochs=20,
            patience=10,
            cuda_device=0 if torch.cuda.is_available() else -1,
            optimizer=optimizer,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps)

    @classmethod
    def _init_weights(cls, module: torch.nn.Module) -> None:
        """Based on Jawahar et al., 2019

        https://github.com/ganeshjawahar/interpret_bert/blob/master/probing/extract_features.py"""
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        if type(module) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        elif type(module) == torch.nn.Embedding:
            torch.nn.init.xavier_uniform_(module.weight)

    def train(self) -> Dict[str, Any]:
        self.model = self._build_model()
        self._trainer = self._build_trainer()
        return self._trainer.train()

    def get_best_weights_path(self) -> str:
        best_weights_path = self._trainer.get_best_weights_path()
        if best_weights_path is None:
            # Use the local serialization dir from the trainer
            best_weights_path = str(Path(self._trainer._serialization_dir) / 'best.th')
        return best_weights_path
