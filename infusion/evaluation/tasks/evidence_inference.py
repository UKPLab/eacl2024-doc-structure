import collections
import dataclasses
import json
import logging
import os
import time
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any, ClassVar

import numpy as np
import omegaconf
from intertext_graph.itgraph import IntertextDocument, Node

from evaluation.common import BaseInstance, BasePrediction, BaseResult, BaseTask, Statistics, Partition
from evaluation.util import precision, recall, f1_score

logger = logging.getLogger(__name__)

LABEL_TO_CODE: Dict[str, int] = {
    "significantly decreased": -1,
    "no significant difference": 0,
    "significantly increased": 1
}

CODE_TO_LABEL: Dict[int, str] = {
    -1: "significantly decreased",
    0: "no significant difference",
    1: "significantly increased"
}


@dataclasses.dataclass
class EvidenceInferenceInstance(BaseInstance):
    pmc_id: str
    prompt_id: str
    # input:
    document: IntertextDocument
    outcome: str
    intervention: str
    comparator: str
    prompt: str
    # output:
    labels: List[str]
    label_codes: List[int]
    evidence_nodes: List[List[Node]]


@dataclasses.dataclass
class EvidenceInferencePrediction(BasePrediction):
    pmc_id: str
    prompt_id: str
    label: str
    label_code: int
    evidence_nodes: List[Node]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "pmc_id": self.pmc_id,
            "prompt_id": self.prompt_id,
            "label": self.label,
            "label_code": self.label_code,
            "evidence_nodes": [node.ix for node in self.evidence_nodes]
        }


@dataclasses.dataclass
class EvidenceInferenceResult(BaseResult):
    classification_class_confusion_matrices: Dict[str, Dict[str, float]]
    classification_class_precisions: Dict[str, float]
    classification_class_recalls: Dict[str, float]
    classification_class_f1_scores: Dict[str, float]
    classification_macro_f1_score: float

    evidence_detection_confusion_matrix: Dict[str, float]
    evidence_detection_precision: float
    evidence_detection_recall: float
    evidence_detection_f1_score: float

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "classification_class_confusion_matrices": self.classification_class_confusion_matrices,
            "classification_class_precisions": self.classification_class_precisions,
            "classification_class_recalls": self.classification_class_recalls,
            "classification_class_f1_scores": self.classification_class_f1_scores,
            "classification_macro_f1_score": self.classification_macro_f1_score,
            "evidence_detection_confusion_matrix": self.evidence_detection_confusion_matrix,
            "evidence_detection_precision": self.evidence_detection_precision,
            "evidence_detection_recall": self.evidence_detection_recall,
            "evidence_detection_f1_score": self.evidence_detection_f1_score,
            "table_entries": self.table_entries
        }

    @property
    def score(self) -> float:
        return self.classification_macro_f1_score + self.evidence_detection_f1_score

    @property
    def table_entries(self) -> Dict[str, float]:
        return {
            "EvI Classification F1": self.classification_macro_f1_score,
            "EvI Evidence F1": self.evidence_detection_f1_score
        }


class EvidenceInferenceTask(BaseTask):
    task_name: ClassVar[str] = "Evidence Inference"

    _train_instances: List[EvidenceInferenceInstance]
    _dev_instances: List[EvidenceInferenceInstance]
    _test_instances: List[EvidenceInferenceInstance]

    def __init__(self, config: omegaconf.DictConfig, stats: Statistics) -> None:
        super(EvidenceInferenceTask, self).__init__(config, stats)

        logger.info("Load the (ITG-formatted) Evidence Inference dataset.")
        tick = time.time()

        self.stats.stats["task-initialization.deep-or-shallow"] = self.config["task"]["deep_or_shallow"]
        train_documents = []
        path = os.path.join(self.config["location"]["datasets"], "EVIDENCE-INFERENCE-ITG",
                            f"{self.config['task']['deep_or_shallow']}-train.jsonl")
        with open(path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    train_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-train-documents"] = len(train_documents)
        dev_documents = []
        path = os.path.join(self.config["location"]["datasets"], "EVIDENCE-INFERENCE-ITG",
                            f"{self.config['task']['deep_or_shallow']}-dev.jsonl")
        with open(path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    dev_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-dev-documents"] = len(dev_documents)
        test_documents = []
        path = os.path.join(self.config["location"]["datasets"], "EVIDENCE-INFERENCE-ITG",
                            f"{self.config['task']['deep_or_shallow']}-test.jsonl")
        with open(path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    test_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-test-documents"] = len(test_documents)

        tack = time.time()
        logger.info(f"Loaded {len(train_documents)} train documents, "
                    f"{len(dev_documents)} dev documents, and "
                    f"{len(test_documents)} test documents in {tack - tick:0.4f}s.")

        logger.info("Create the Evidence Inference instances.")
        tick = time.time()

        self.stats.stats["task-initialization.prompt-has-no-annotations"] = 0
        self.stats.stats["task-initialization.annotation-has-empty-evidence-text"] = 0
        self.stats.stats["task-initialization.annotation-has-no-evidence-node"] = 0
        self.stats.stats["task-initialization.annotation-has-more-than-one-evidence-node"] = 0

        self._train_instances = []
        for document in train_documents:
            self._train_instances += self._create_train_instances_from_document(document)
        self.stats.stats["task-initialization.num-train-instances"] = len(self.train_instances)

        self._dev_instances = []
        for document in dev_documents:
            self._dev_instances += self._create_eval_instances_from_document(document)
        self.stats.stats["task-initialization.num-dev-instances"] = len(self.dev_instances)

        self._test_instances = []
        for document in test_documents:
            self._test_instances += self._create_eval_instances_from_document(document)
        self.stats.stats["task-initialization.num-test-instances"] = len(self.test_instances)

        tack = time.time()
        logger.info(f"Created {len(self.train_instances)} train instances, "
                    f"{len(self.dev_instances)} dev instances, and "
                    f"{len(self.test_instances)} test instances in {tack - tick:0.4f}s.")

        logger.info("Gather label and evidence statistics.")
        tick = time.time()
        self.stats.stats["task-initialization.label-statistics"] = {-1: 0, 0: 0, 1: 0}
        self.stats.stats["task-initialization.evidence-statistics"] = {0: 0, 1: 0}

        for instance in self._train_instances:
            self.stats.stats["task-initialization.label-statistics"][instance.label_codes[0]] += 1
            num_nodes = len(instance.document.nodes)
            num_evidence_nodes = len(instance.evidence_nodes[0])
            self.stats.stats["task-initialization.evidence-statistics"][0] += num_nodes - num_evidence_nodes
            self.stats.stats["task-initialization.evidence-statistics"][1] += num_evidence_nodes

        logger.info(f"Label statistics: {self.stats.stats['task-initialization.label-statistics']}")
        logger.info(f"Evidence statistics: {self.stats.stats['task-initialization.evidence-statistics']}")

        tack = time.time()
        logger.info(f"Gathered label and evidence statistics in {tack - tick:0.4f}s.")

    def evaluate(self, instances: List[EvidenceInferenceInstance],
                 predictions: List[EvidenceInferencePrediction], partition: Partition) -> EvidenceInferenceResult:
        # TODO: align this with evaluation in evidence inference paper
        # make predictions and fill confusion matrices
        class_confusion = {
            label: collections.Counter() for label in LABEL_TO_CODE.keys()
        }

        evidence_confusion = collections.Counter()

        for instance, prediction in zip(instances, predictions):
            # classification: evaluate for each ground truth and choose the best one
            class_confusions_here = []
            scores_here = []
            for true_label in instance.labels:
                class_confusion_here = {
                    label: collections.Counter() for label in LABEL_TO_CODE.keys()
                }
                for label in LABEL_TO_CODE.keys():
                    if true_label == label and prediction.label == label:
                        class_confusion_here[label]["TP"] += 1
                    elif true_label == label and prediction.label != label:
                        class_confusion_here[label]["FN"] += 1
                    elif true_label != label and prediction.label == label:
                        class_confusion_here[label]["FP"] += 1
                    else:
                        class_confusion_here[label]["TN"] += 1
                score_here = sum(class_confusion_here[label]["TP"] + class_confusion_here[label]["TN"] for label in LABEL_TO_CODE.keys())

                class_confusions_here.append(class_confusion_here)
                scores_here.append(score_here)

            # choose the 'best-fitting' ground truth/evaluation
            best_idx = scores_here.index(max(scores_here))
            best_class_confusion_here = class_confusions_here[best_idx]
            for label in LABEL_TO_CODE.keys():
                class_confusion[label] += best_class_confusion_here[label]

            # evidence detection: evaluate for each ground truth and choose the best one
            evidence_confusions_here = []
            scores_here = []

            # TODO: maybe filter out cases in which the evidence ground truth has errors (e.g., no nodes found)
            for evidence_nodes in instance.evidence_nodes:
                evidence_confusion_here = collections.Counter()
                true_ixs = set(node.ix for node in evidence_nodes)
                pred_ixs = set(node.ix for node in prediction.evidence_nodes)
                for node in instance.document.nodes:
                    if node.ix in true_ixs and node.ix in pred_ixs:
                        evidence_confusion_here["TP"] += 1
                    elif node.ix in true_ixs and node.ix not in pred_ixs:
                        evidence_confusion_here["FN"] += 1
                    elif node.ix not in true_ixs and node.ix in pred_ixs:
                        evidence_confusion_here["FP"] += 1
                    else:
                        evidence_confusion_here["TN"] += 1
                score_here = evidence_confusion_here["TP"] + evidence_confusion_here["TN"]

                evidence_confusions_here.append(evidence_confusion_here)
                scores_here.append(score_here)

            # choose the 'best-fitting' ground truth/evaluation
            best_idx = scores_here.index(max(scores_here))
            best_evidence_confusion_here = evidence_confusions_here[best_idx]
            evidence_confusion += best_evidence_confusion_here

        # compute precisions, recalls, and F1 scores for classification
        class_precisions = {}
        class_recalls = {}
        class_f1_scores = {}
        for label in LABEL_TO_CODE.keys():
            class_precisions[label] = precision(class_confusion[label])
            class_recalls[label] = recall(class_confusion[label])
            class_f1_scores[label] = f1_score(class_confusion[label])
        macro_f1_score = float(np.mean(np.array([class_f1_scores[label] for label in LABEL_TO_CODE.keys()])))

        # compute precision, recall, and F1 score for evidence detection
        evidence_precision = precision(evidence_confusion)
        evidence_recall = recall(evidence_confusion)
        evidence_f1_score = f1_score(evidence_confusion)

        return EvidenceInferenceResult(
            classification_class_confusion_matrices={k: dict(v) for k, v in class_confusion.items()},
            classification_class_precisions=class_precisions,
            classification_class_recalls=class_recalls,
            classification_class_f1_scores=class_f1_scores,
            classification_macro_f1_score=macro_f1_score,
            evidence_detection_confusion_matrix=evidence_confusion,
            evidence_detection_precision=evidence_precision,
            evidence_detection_recall=evidence_recall,
            evidence_detection_f1_score=evidence_f1_score
        )

    @property
    def train_instances(self) -> List[EvidenceInferenceInstance]:
        return self._train_instances

    @property
    def dev_instances(self) -> List[EvidenceInferenceInstance]:
        return self._dev_instances

    @property
    def test_instances(self) -> List[EvidenceInferenceInstance]:
        return self._test_instances

    def _find_evidence_nodes(self, document: IntertextDocument, annotation: Dict[str, Any]) -> List[Node]:
        evidence_nodes = []
        if annotation["evidence_text"] == "":
            self.stats.stats["task-initialization.annotation-has-empty-evidence-text"] += 1
            return evidence_nodes
        for node in document.nodes:
            for evidence_for in node.meta["is_evidence_for"]:
                if evidence_for["prompt_id"] == annotation["prompt_id"] and evidence_for["user_id"] == annotation["user_id"]:
                    evidence_nodes.append(node)
                    break
        if len(evidence_nodes) == 0:
            self.stats.stats["task-initialization.annotation-has-no-evidence-node"] += 1
        elif len(evidence_nodes) > 1:
            self.stats.stats["task-initialization.annotation-has-more-than-one-evidence-node"] += 1
        return evidence_nodes

    def _create_train_instances_from_document(self, doc: IntertextDocument) -> List[EvidenceInferenceInstance]:
        instances = []

        # create multiple instances per prompt (one for each annotation)
        for prompt_id, prompt in doc.meta["prompts"].items():
            if prompt_id not in doc.meta["annotations"].keys():
                self.stats.stats["task-initialization.prompt-has-no-annotations"] += 1
                continue

            for annotation in doc.meta["annotations"][prompt_id]:
                evidence_nodes = self._find_evidence_nodes(doc, annotation)
                instance = EvidenceInferenceInstance(
                    pmc_id=doc.meta["pmc_id"],
                    prompt_id=prompt_id,
                    document=doc,
                    outcome=prompt["outcome"],
                    intervention=prompt["intervention"],
                    comparator=prompt["comparator"],
                    prompt=self._create_prompt(prompt["outcome"], prompt["intervention"], prompt["comparator"]),
                    labels=[annotation["label"]],
                    label_codes=[annotation["label_code"]],
                    evidence_nodes=[evidence_nodes]
                )
                instances.append(instance)

        return instances

    def _create_eval_instances_from_document(self, doc: IntertextDocument) -> List[EvidenceInferenceInstance]:
        instances = []

        # create one instance per prompt (with all annotations)
        for prompt_id, prompt in doc.meta["prompts"].items():
            if prompt_id not in doc.meta["annotations"].keys():
                self.stats.stats["task-initialization.prompt-has-no-annotations"] += 1
                continue

            all_labels = []
            all_label_codes = []
            all_evidence_nodes = []

            for annotation in doc.meta["annotations"][prompt_id]:
                evidence_nodes = self._find_evidence_nodes(doc, annotation)

                all_label_codes.append(annotation["label_code"])
                all_labels.append(annotation["label"])
                all_evidence_nodes.append(evidence_nodes)

            instance = EvidenceInferenceInstance(
                pmc_id=doc.meta["pmc_id"],
                prompt_id=prompt_id,
                document=doc,
                outcome=prompt["outcome"],
                intervention=prompt["intervention"],
                comparator=prompt["comparator"],
                prompt=self._create_prompt(prompt["outcome"], prompt["intervention"], prompt["comparator"]),
                labels=all_labels,
                label_codes=all_label_codes,
                evidence_nodes=all_evidence_nodes
            )
            instances.append(instance)

        return instances

    def _create_prompt(self, outcome: str, intervention: str, comparator: str) -> str:
        return f"With respect to {outcome}, characterize the reported difference" \
               f" between patients receiving {intervention} and those receiving {comparator}."

    @staticmethod
    def load_predictions(
            input_path: Path,
            instances: List[EvidenceInferenceInstance]
    ) -> List[EvidenceInferencePrediction]:
        predictions = []
        instance_map = {
            f'{instance.pmc_id}_{instance.prompt_id}': instance for instance in instances
        }
        with open(input_path, "r") as f:
            for line in f:
                prediction_dict = json.loads(line)
                instance = instance_map[f'{prediction_dict["pmc_id"]}_{prediction_dict["prompt_id"]}']
                evidence_nodes = [
                    instance.document.get_node_by_ix(node_id)
                    for node_id in prediction_dict["evidence_nodes"]
                ]
                predictions.append(EvidenceInferencePrediction(
                    pmc_id=prediction_dict["pmc_id"],
                    prompt_id=prediction_dict["prompt_id"],
                    label=prediction_dict["label"],
                    label_code=prediction_dict["label_code"],
                    evidence_nodes=evidence_nodes
                ))
        # sort predictions according to instance order
        predictions.sort(key=lambda p: instances.index(instance_map[f'{p.pmc_id}_{p.prompt_id}']))
        return predictions
