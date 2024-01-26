import abc
import dataclasses
import enum
import hashlib
import logging
import os
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List, ClassVar, Tuple, Optional, Union

import omegaconf
import pytorch_lightning as pl
import torch
from intertext_graph.itgraph import IntertextDocument
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


################################################################################
# task-specific
################################################################################

@dataclasses.dataclass
class BaseInstance(abc.ABC):
    """Base class for all dataset instances."""
    pass


@dataclasses.dataclass
class BasePrediction(abc.ABC):
    """Base class for all dataset predictions."""
    pass


@dataclasses.dataclass
class BaseResult(abc.ABC):
    """Base class for all dataset evaluation results."""

    @abc.abstractmethod
    def to_json_dict(self) -> Dict[str, Any]:
        """Return the result object as a JSON-serializable dictionary."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def score(self) -> float:
        """Score that determines what is considered as 'better' during fine-tuning."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def table_entries(self) -> Dict[str, float]:
        """Result values that should appear in the overall results table."""
        raise NotImplementedError


class BaseTask(abc.ABC):
    """Base class for all downstream evaluation tasks."""
    task_name: ClassVar[str] = "BaseTask"

    config: omegaconf.DictConfig
    stats: "Statistics"

    def __init__(self, config: omegaconf.DictConfig, stats: "Statistics") -> None:
        super(BaseTask, self).__init__()
        assert config["task"]["task_name"] == self.task_name
        self.config = config
        self.stats = stats

    @abc.abstractmethod
    def evaluate(self, instances: List[BaseInstance],
                 predictions: List[BasePrediction], partition: "Partition") -> BaseResult:
        """Evaluate the predictions on the downstream task. Implemented by each individual task."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def train_instances(self) -> List[BaseInstance]:
        """Train instances of the dataset. Implemented by each individual task."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dev_instances(self) -> List[BaseInstance]:
        """Dev instances of the dataset. Implemented by each individual task."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def test_instances(self) -> List[BaseInstance]:
        """Test instances of the dataset. Implemented by each individual task."""
        raise NotImplementedError

    @staticmethod
    def save_predictions(
            predictions: List[BasePrediction],
            output_path: Path
    ) -> None:
        """Save the predictions to disk."""

        with open(output_path, "w") as f:
            for prediction in predictions:
                f.write(json.dumps(prediction.to_json_dict(), indent=None))
                f.write('\n')

    @staticmethod
    @abc.abstractmethod
    def load_predictions(
            input_path: Path,
            instances: List[BaseInstance]
    ) -> List[BasePrediction]:
        """Load the predictions from disk. Implemented by each individual task."""
        raise NotImplementedError



################################################################################
# model-specific
################################################################################

class BaseModel(pl.LightningModule, abc.ABC):
    model_name: ClassVar[str] = "BaseModel"
    task_name: ClassVar[str] = "BaseTask"

    config: omegaconf.DictConfig
    stats: "Statistics"

    def __init__(self, config: Optional[omegaconf.DictConfig], stats: Optional["Statistics"]) -> None:
        super(BaseModel, self).__init__()
        assert config["model"]["model_name"] == self.model_name
        assert config["task"]["task_name"] == self.task_name
        self.config = config
        self.stats = stats

    @abc.abstractmethod
    def training_collate_fn(self, instances: List[BaseInstance]) -> Any:
        """
        Collate function that creates the training/evaluation batches.
        Implemented by each individual model wrapper.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validation_collate_fn(self, instances: List[BaseInstance]) -> Any:
        """
        Collate function that creates the training/evaluation batches.
        Implemented by each individual model wrapper.
        """
        raise NotImplementedError

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def get_batch_size_and_accumulation_steps(
            self,
            batch_size: int,
            gradient_accumulation_steps: int
    ) -> Tuple[int, int]:
        if not torch.cuda.is_available():
            logger.info('No GPU available, returning given values')
            return batch_size, gradient_accumulation_steps

        available_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        logger.info(f'Available GPU memory: {available_gpu_memory}')
        num_accumulated_instances = batch_size * gradient_accumulation_steps

        if self.model_name in ['LED', 'LongT5']:
            if available_gpu_memory < 52e9:
                # 40GB GPU or 48GB GPU
                batch_size = 1
            else:
                # 80GB GPU
                batch_size = 2
            gradient_accumulation_steps = num_accumulated_instances // batch_size

        else:
            raise NotImplementedError

        return batch_size, gradient_accumulation_steps


    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # checkpoint is dictionary, need to resize individual matrices
        # or resize the model (self)

        resized_parameter_names = []

        if self.model_name == 'LED':

            if self.task_name in [
                'QASPER',
                'Scaffold Task For Pretraining'
            ]:
                # hard-code the names of the led parameters that change size when calling
                # resize_token_embeddings()
                resized_parameter_names = [
                    'led_model.final_logits_bias',
                    'led_model.led.shared.weight',
                    'led_model.led.encoder.embed_tokens.weight',
                    'led_model.led.decoder.embed_tokens.weight',
                    'led_model.lm_head.weight'
                ]

            elif self.task_name in ['Evidence Inference']:
                # hard-code the names of the led parameters that change size when calling
                # resize_token_embeddings()
                resized_parameter_names = [
                    'led_model.shared.weight',
                    'led_model.encoder.embed_tokens.weight',
                    'led_model.decoder.embed_tokens.weight'
                ]

                test_key = [
                    k for k in checkpoint['state_dict'].keys()
                ][1]
                if '.led.' in test_key:
                    # Because Evidence Inference uses
                    # transformers.LEDModel the parameter naming is different when
                    # loading the checkpoint of a LEDForConditionalGeneration model
                    # The parameters in the loaded checkpoint need to be renamed
                    # Remove the '.led' in the parameter names of the checkpoint
                    new_state_dict = {}
                    for k, v in checkpoint['state_dict'].items():
                        if k.startswith('led_model.led.'):
                            new_k = f'led_model.{".".join([part for part in k.split(".")[2:]])}'
                            new_state_dict[new_k] = v
                        else:
                            new_state_dict[k] = v

                    checkpoint['state_dict'] = OrderedDict(new_state_dict)

            else:
                raise NotImplementedError

        elif self.model_name == 'LongT5':
            if self.task_name in [
                'QASPER',
                'Scaffold Task For Pretraining',
                'Evidence Inference',
            ]:
                resized_parameter_names = [
                    'model.shared.weight',
                    'model.encoder.embed_tokens.weight',
                    'model.decoder.embed_tokens.weight',
                    'model.lm_head.weight'
                ]

        for param_name in resized_parameter_names:
            # When the loaded state dict belongs to a model that was trained with
            # t5-style denoising pretraining, it has additional parameters that
            # correspond to sentinel tokens. These additional parameters are removed
            # here to ensure that the tensor sizes in the model and the loaded state
            # dict are the same
            if (
                    checkpoint['state_dict'][param_name].size()
                    > self.state_dict()[param_name].size()
            ):
                target_shape = self.state_dict()[param_name].size()

                if len(target_shape) == 1:
                    new_param = checkpoint['state_dict'][param_name][:target_shape[0]]

                elif len(target_shape) == 2:
                    new_param = checkpoint['state_dict'][param_name][
                                :target_shape[0], :target_shape[1]
                                ]
                elif len(target_shape) == 3:
                    new_param = checkpoint['state_dict'][param_name][
                                :target_shape[0], :target_shape[1], :target_shape[2]
                                ]
                else:
                    raise NotImplementedError(
                        f'Parameter {param_name} has {len(target_shape)} dimensions, '
                        f'which is more than expected.'
                    )

                checkpoint['state_dict'][param_name] = new_param



################################################################################
# evaluation
################################################################################

class Partition(enum.Enum):
    """Partition of the dataset."""
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@dataclasses.dataclass
class Statistics:
    """Statistics object for statistics gathered throughout the evaluation process."""
    model_name: str
    task_name: str
    description: str

    config: omegaconf.DictConfig

    stats: Dict[str, Any] = dataclasses.field(init=False, default_factory=dict)

    results_by_step: List[Tuple[int, BaseResult]] = dataclasses.field(init=False, default_factory=list)
    best_num_steps: int = dataclasses.field(init=False, default=0)
    test_result: BaseResult = dataclasses.field(init=False)

    total_time: float = dataclasses.field(init=False, default=0.0)

    def to_json_dict(self) -> Dict[str, Any]:
        """Return the statistics object as a JSON-serializable dictionary."""
        return {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "description": self.description,
            "config": omegaconf.OmegaConf.to_object(self.config),
            "hash": self.to_hash(),
            "stats": self.stats,
            "results_by_step": [(n, r.to_json_dict()) for n, r in self.results_by_step],
            "best_num_steps": self.best_num_steps,
            "test_result": self.test_result.to_json_dict(),
            "total_time": self.total_time
        }

    def to_hash(self) -> str:
        h = hashlib.sha256(bytes(f"{self.model_name}{self.task_name}"
                                 f"{self.description}{dict(self.config)}", "utf-8")).hexdigest()
        return f"{h[:4]}-{h[4:8]}"


class CustomDataset(Dataset):
    """Simple generic PyTorch dataset implementation."""
    instances: Union[List[BaseInstance], Dataset]

    def __init__(self, instances: List[BaseInstance]) -> None:
        super(CustomDataset, self).__init__()
        self.instances = instances

    def __getitem__(self, item) -> BaseInstance:
        return self.instances[item]

    def __len__(self) -> int:
        return len(self.instances)


class SingleFileDataset(Dataset):
    """Dataset that loads instances from single ITG files. """
    dir_path: Path
    valid_filenames: List = None

    def __init__(
            self,
            dir_path: Path,
            valid_filenames: List = None
    ) -> None:
        super(SingleFileDataset, self).__init__()
        self.dir_path = dir_path

        if valid_filenames is None or len(valid_filenames) == 0:
            self.filenames = os.listdir(self.dir_path)
        else:
            self.filenames = [
                filename for filename in os.listdir(self.dir_path)
                if filename in valid_filenames
            ]

    def __getitem__(self, item) -> BaseInstance:
        with open(self.dir_path / self.filenames[item], 'r') as file:
            document = IntertextDocument.load_json(file)

        instance = self._create_instance_from_document(document)

        return instance

    def __len__(self) -> int:
        return len(self.filenames)

    @staticmethod
    @abc.abstractmethod
    def _create_instance_from_document(document: IntertextDocument) -> BaseInstance:
        raise NotImplementedError