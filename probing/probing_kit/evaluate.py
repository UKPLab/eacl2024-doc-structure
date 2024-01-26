from __future__ import annotations
from typing import Dict, Tuple

import torch

from probing_kit.classifier import ProbingModel


class EvaluationForwardHook:

    def __init__(self, model: ProbingModel) -> None:
        self._model = model
        self.predictions = []

    def __enter__(self) -> EvaluationForwardHook:
        self._handle = self._model.register_forward_hook(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._handle.remove()

    def __call__(self, model: ProbingModel, input: Tuple, output: Dict[str, torch.Tensor]) -> None:
        human_readable = model.make_output_human_readable(output)
        self.predictions += human_readable['labels']
