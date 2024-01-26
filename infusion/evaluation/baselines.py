import logging
import random
from typing import Any, List, ClassVar, Tuple

import omegaconf
import torch
import torch.nn.functional as F

from evaluation.common import BaseModel, Statistics, BaseInstance
from evaluation.tasks.evidence_inference import EvidenceInferencePrediction, EvidenceInferenceTask, EvidenceInferenceInstance
from evaluation.tasks.qasper import QASPERTask, QASPERPrediction

logger = logging.getLogger(__name__)


class Batch:
    instances: List[BaseInstance]

    def __init__(self, instances) -> None:
        self.instances = instances


class OracleForEvidenceInferenceModel(BaseModel):
    model_name: ClassVar[str] = "Oracle (ground truth)"
    task_name: ClassVar[str] = EvidenceInferenceTask.task_name

    dummy_layer: torch.nn.Linear

    def __init__(self, config: omegaconf.DictConfig, stats: Statistics) -> None:
        super(OracleForEvidenceInferenceModel, self).__init__(config, stats)
        self.dummy_layer = torch.nn.Linear(
            in_features=3,
            out_features=2
        )

    def training_step(self, batch, batch_idx):
        tmp = torch.tensor([[0.1, 1.2, 2.3]], device=self.device)
        tmp = self.dummy_layer(tmp)
        loss = F.cross_entropy(tmp, torch.tensor([1], device=self.device))
        return loss

    def validation_step(self, batch, batch_idx):
        predictions = []
        for instance in batch:
            idx = random.randint(0, len(instance.labels) - 1)
            prediction = EvidenceInferencePrediction(
                label=instance.labels[idx],
                label_code=instance.label_codes[idx],
                evidence_nodes=instance.evidence_nodes[idx]
            )
            predictions.append(prediction)
        return predictions

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def collate_fn(self, instances: List[EvidenceInferenceInstance]) -> Any:
        return instances


class OracleForQASPERModel(BaseModel):
    model_name: ClassVar[str] = "Oracle (ground truth)"
    task_name: ClassVar[str] = QASPERTask.task_name

    dummy_layer: torch.nn.Linear

    def __init__(self, config: omegaconf.DictConfig, stats: Statistics) -> None:
        super(OracleForQASPERModel, self).__init__(config, stats)
        self.dummy_layer = torch.nn.Linear(
            in_features=3,
            out_features=2
        )

    def training_step(self, batch, batch_idx):
        tmp = torch.tensor([[0.1, 1.2, 2.3]], device=self.device)
        tmp = self.dummy_layer(tmp)
        loss = F.cross_entropy(tmp, torch.tensor([1], device=self.device))
        return loss

    def validation_step(self, batch, batch_idx):
        predictions = []
        for instance in batch:
            # make sure that for any of the annotations of the prompt, the oracle works perfectly
            idx = random.randint(0, len(instance.answer_texts) - 1)
            prediction = QASPERPrediction(
                question_id=instance.question_id,
                answer_text=instance.answer_texts[idx],
                evidence_nodes=instance.evidence_nodes[idx]
            )
            predictions.append(prediction)
        return predictions

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def collate_fn(self, instances: List[EvidenceInferenceInstance]) -> Any:
        return instances
