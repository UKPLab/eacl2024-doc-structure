import dataclasses
import json
import logging
import math
import os
import time
from io import StringIO
from pathlib import Path
from random import random
from typing import List, Dict, Any, ClassVar

import numpy as np
import omegaconf
from intertext_graph.itgraph import IntertextDocument

from evaluation.common import BaseInstance, BasePrediction, BaseResult, BaseTask, Statistics, Partition
from structformer.scaffold_tasks import S2ORCITGSubsetDataHandler

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ScaffoldTaskForPretrainingInstance(BaseInstance):
    # input:
    document: IntertextDocument


@dataclasses.dataclass
class ScaffoldTaskForPreTrainingPrediction(BasePrediction):
    loss: float


@dataclasses.dataclass
class ScaffoldTaskForPretrainingResult(BaseResult):
    loss: float

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
        }

    @property
    def score(self) -> float:
        # return negative loss because we need a score for which
        # bigger is better
        return - self.loss

    @property
    def table_entries(self) -> Dict[str, float]:
        return {
            "Scaffold Loss": self.loss
        }


class ScaffoldTaskForPretraining(BaseTask):
    task_name: ClassVar[str] = "Scaffold Task For Pretraining"

    _train_instances: List[ScaffoldTaskForPretrainingInstance]
    _dev_instances: List[ScaffoldTaskForPretrainingInstance]
    _test_instances: List[ScaffoldTaskForPretrainingInstance]

    def __init__(self, config: omegaconf.DictConfig, stats: Statistics) -> None:
        super(ScaffoldTaskForPretraining, self).__init__(config, stats)

        logger.info("Load the (ITG-formatted) Scaffold Task dataset.")
        tick = time.time()

        # store the used datasets
        self.stats.stats['datasets'] = list(self.config['task']['datasets'])

        self.stats.stats["task-initialization.deep-or-shallow"] = None
        documents = {
            'train': [],
            'dev': [],
            'test': []
        }
        for dataset_name in self.config['task']['datasets']:
            if 'S2ORC' in dataset_name:
                # This abuses the S2ORCITGSubsetDataHandler a little, as it returns
                # S2ORCITGInstances, from which we only need the document
                # Also, we set the num_docs_per_shard in the config['scaffold_tasks'] attribute
                # of the config (and not in the config['tasks'] attribute as we do it for the
                # rest of the configurations for the ScaffoldTask
                s2orc_instances = S2ORCITGSubsetDataHandler(
                    self.config,
                    self.stats
                ).instances
                s2orc_docs = [
                    instance.document
                    for instance in s2orc_instances
                ]
                del s2orc_instances

                split_proportions = (0.7, 0.1, 0.2)
                first_dev_doc = math.floor(split_proportions[0] * len(s2orc_docs))
                first_test_doc = math.floor(sum(split_proportions[:2]) * len(s2orc_docs))
                documents['train'].extend(s2orc_docs[:first_dev_doc])
                documents['dev'].extend(s2orc_docs[first_dev_doc:first_test_doc])
                documents['test'].extend(s2orc_docs[first_test_doc:])

            elif 'F1000' in dataset_name:
                for split_name in documents.keys():
                    for filename in os.listdir(
                        self.config['location']['datasets'] + '/' + dataset_name + '/' + split_name
                    ):
                        filepath = os.path.join(
                            self.config['location']['datasets'],
                            dataset_name,
                            split_name,
                            filename
                        )
                        with open(filepath) as file:
                            itg = IntertextDocument.load_json(file)

                        documents[split_name].append(itg)

            else:
                for split_name in documents.keys():
                    loaded_docs = self._load_regular_dataset(
                        self.config['location']['datasets'],
                        dataset_name,
                        split_name
                    )
                    documents[split_name].extend(loaded_docs)



        self.stats.stats["task-initialization.num-train-documents"] = len(documents['train'])
        self.stats.stats["task-initialization.num-dev-documents"] = len(documents['dev'])
        self.stats.stats["task-initialization.num-test-documents"] = len(documents['test'])

        tack = time.time()
        logger.info(f"Loaded {len(documents['train'])} train documents, "
                    f"{len(documents['dev'])} dev documents, and "
                    f"{len(documents['test'])} test documents in {tack - tick:0.4f}s.")

        logger.info("Create the Scaffold Task instances.")
        tick = time.time()

        self._train_instances = []
        for document in documents['train']:
            self._train_instances += self._create_train_instances_from_document(document)
        self.stats.stats["task-initialization.num-train-instances"] = len(self.train_instances)

        self._dev_instances = []
        for document in documents['dev']:
            self._dev_instances += self._create_eval_instances_from_document(document)
        self.stats.stats["task-initialization.num-dev-instances"] = len(self.dev_instances)

        self._test_instances = []
        for document in documents['test']:
            self._test_instances += self._create_eval_instances_from_document(document)
        self.stats.stats["task-initialization.num-test-instances"] = len(self.test_instances)

        tack = time.time()
        logger.info(f"Created {len(self.train_instances)} train instances, "
                    f"{len(self.dev_instances)} dev instances, and "
                    f"{len(self.test_instances)} test instances in {tack - tick:0.4f}s.")


    def evaluate(self, instances: List[ScaffoldTaskForPretrainingInstance],
                 predictions: List[ScaffoldTaskForPreTrainingPrediction], partition: Partition) -> ScaffoldTaskForPretrainingResult:

        mean_loss = np.mean(
            [
                prediction.loss
                for prediction in predictions
            ]
        )

        return ScaffoldTaskForPretrainingResult(mean_loss)

    @property
    def train_instances(self) -> List[ScaffoldTaskForPretrainingInstance]:
        return self._train_instances

    @property
    def dev_instances(self) -> List[ScaffoldTaskForPretrainingInstance]:
        return self._dev_instances

    @property
    def test_instances(self) -> List[ScaffoldTaskForPretrainingInstance]:
        return self._test_instances

    @staticmethod
    def _create_train_instances_from_document(doc: IntertextDocument) -> List[ScaffoldTaskForPretrainingInstance]:
        instances = [ScaffoldTaskForPretrainingInstance(doc)]

        return instances

    @staticmethod
    def _create_eval_instances_from_document(doc: IntertextDocument) -> List[ScaffoldTaskForPretrainingInstance]:
        instances = [ScaffoldTaskForPretrainingInstance(doc)]

        return instances

    @staticmethod
    def _load_regular_dataset(
            datasets_path: str,
            dataset_name: str,
            split: str = None
    ) -> List[IntertextDocument]:
        documents = []
        if dataset_name.endswith('/'):
            connector = ''
        else:
            connector = '-'
        path = os.path.join(
            datasets_path,
            f'{dataset_name}{connector}{split}.jsonl'
        )
        with open(path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    documents.append(IntertextDocument.load_json(f))

        return documents

    def load_predictions(
            input_path: Path,
            instances: List[BaseInstance]
    ) -> List[BasePrediction]:
        raise NotImplementedError()