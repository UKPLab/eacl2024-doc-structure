import collections
import dataclasses
import logging
import os
import re
import string
import time
import json
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any, ClassVar, Tuple

import omegaconf
from intertext_graph.itgraph import IntertextDocument, Node
from rouge_score.rouge_scorer import RougeScorer

from evaluation.common import BaseInstance, BasePrediction, BaseResult, BaseTask, Statistics, Partition

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class QASPERInstance(BaseInstance):
    question_id: str

    # input:
    document: IntertextDocument
    question: str

    # output:
    answer_texts: List[str]
    evidence_nodes: List[List[Node]]


@dataclasses.dataclass
class QASPERPrediction(BasePrediction):
    question_id: str

    answer_text: str
    evidence_nodes: List[Node]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "answer_text": self.answer_text,
            "evidence_nodes": [node.ix for node in self.evidence_nodes]
        }


@dataclasses.dataclass
class QASPERResult(BaseResult):
    answer_f1: float
    answer_f1_by_type: Dict[str, float]
    evidence_f1: float
    num_missing_predictions: int

    multi_mean_rouge_1: float
    multi_mean_rouge_2: float
    multi_mean_rouge_l: float

    multi_max_rouge_1: float
    multi_max_rouge_2: float
    multi_max_rouge_l: float

    samples: List[Tuple[List[str], str]]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "answer_f1": self.answer_f1,
            "answer_f1_by_type": self.answer_f1_by_type,
            "evidence_f1": self.evidence_f1,
            "num_missing_predictions": self.num_missing_predictions,
            "multi_mean_rouge_1": self.multi_mean_rouge_1,
            "multi_mean_rouge_2": self.multi_mean_rouge_2,
            "multi_mean_rouge_l": self.multi_mean_rouge_l,
            "multi_max_rouge_1": self.multi_max_rouge_1,
            "multi_max_rouge_2": self.multi_max_rouge_2,
            "multi_max_rouge_l": self.multi_max_rouge_l,
            "samples": self.samples,
            "table_entries": self.table_entries
        }

    @property
    def score(self) -> float:
        return self.answer_f1 + self.evidence_f1

    @property
    def table_entries(self) -> Dict[str, float]:
        return {
            "QASPER Answer F1": self.answer_f1,
            "QASPER Evidence F1": self.evidence_f1,
            "QASPER R1": self.multi_max_rouge_1,
            "QASPER R2": self.multi_max_rouge_2,
            "QASPER RL": self.multi_max_rouge_l
        }


class QASPERTask(BaseTask):
    task_name: ClassVar[str] = "QASPER"

    train_documents: List[IntertextDocument]
    dev_documents: List[IntertextDocument]
    test_documents: List[IntertextDocument]

    _train_instances: List[QASPERInstance]
    _dev_instances: List[QASPERInstance]
    _test_instances: List[QASPERInstance]

    def __init__(self, config: omegaconf.DictConfig, stats: Statistics) -> None:
        super(QASPERTask, self).__init__(config, stats)

        logger.info("Load the QASPER-ITG dataset.")
        tick = time.time()

        deep_or_shallow = self.config['task']['deep_or_shallow']

        train_path = os.path.join(self.config["location"]["datasets"], "QASPER-ITG", f"{deep_or_shallow}-train.jsonl")
        self.train_documents = []
        with open(train_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    self.train_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-train-documents"] = len(self.train_documents)

        dev_path = os.path.join(self.config["location"]["datasets"], "QASPER-ITG", f"{deep_or_shallow}-dev.jsonl")
        self.dev_documents = []
        with open(dev_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    self.dev_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-dev-documents"] = len(self.dev_documents)

        test_path = os.path.join(self.config["location"]["datasets"], "QASPER-ITG", f"{deep_or_shallow}-test.jsonl")
        self.test_documents = []
        with open(test_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    self.test_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-test-documents"] = len(self.test_documents)

        tack = time.time()
        logger.info(f"Loaded {len(self.train_documents)} train documents, "
                    f"{len(self.dev_documents)} dev documents, and "
                    f"{len(self.test_documents)} test documents in {tack - tick:0.4f}s.")

        logger.info("Create the QASPER instances.")
        tick = time.time()

        self._train_instances = []
        for document in self.train_documents:
            self._train_instances += self._create_train_instances_from_document(document)
        self.stats.stats["task-initialization.num-train-instances"] = len(self.train_instances)

        self._dev_instances = []
        for document in self.dev_documents:
            self._dev_instances += self._create_eval_instances_from_document(document)
        self.stats.stats["task-initialization.num-dev-instances"] = len(self.dev_instances)

        self._test_instances = []
        for document in self.test_documents:
            self._test_instances += self._create_eval_instances_from_document(document)
        self.stats.stats["task-initialization.num-test-instances"] = len(self.test_instances)

        tack = time.time()
        logger.info(f"Created {len(self.train_instances)} train instances, "
                    f"{len(self.dev_instances)} dev instances, and "
                    f"{len(self.test_instances)} test instances in {tack - tick:0.4f}s.")

        logger.info("Gather evidence statistics.")
        tick = time.time()
        self.stats.stats["task-initialization.evidence-statistics"] = {0: 0, 1: 0}

        for instance in self._train_instances:
            num_nodes = len(instance.document.nodes)
            num_evidence_nodes = len(instance.evidence_nodes[0])
            self.stats.stats["task-initialization.evidence-statistics"][0] += num_nodes - num_evidence_nodes
            self.stats.stats["task-initialization.evidence-statistics"][1] += num_evidence_nodes

        logger.info(f"Evidence statistics: {self.stats.stats['task-initialization.evidence-statistics']}")

        tack = time.time()
        logger.info(f"Gathered evidence statistics in {tack - tick:0.4f}s.")

    def evaluate(self, instances: List[QASPERInstance],
                 predictions: List[QASPERPrediction], partition: Partition) -> QASPERResult:
        # The evaluation is based on QASPER's own evaluator and copies from its code
        # see https://github.com/allenai/qasper-led-baseline/blob/afd0fb96bf78ce8cd8157639c6f6a6995e4f9089/scripts/evaluator.py
        def normalize_answer(s):
            """
            Taken from the official evaluation script for v1.1 of the SQuAD dataset.
            Lower text and remove punctuation, articles and extra whitespace.
            """

            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)

            def white_space_fix(text):
                return " ".join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(remove_punc(lower(s))))

        def token_f1_score(prediction, ground_truth):
            """
            Taken from the official evaluation script for v1.1 of the SQuAD dataset.
            """
            prediction_tokens = normalize_answer(prediction).split()
            ground_truth_tokens = normalize_answer(ground_truth).split()
            common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        def paragraph_f1_score(prediction, ground_truth):
            if not ground_truth and not prediction:
                # The question is unanswerable and the prediction is empty.
                return 1.0
            num_same = len(set(ground_truth).intersection(set(prediction)))
            if num_same == 0:
                return 0.0
            precision = num_same / len(prediction)
            recall = num_same / len(ground_truth)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        # this function is slightly adapted to receive intertext documents as input
        # def get_answers_and_evidence(data, text_evidence_only):
        #     answers_and_evidence = {}
        #     for paper_data in data.values():
        #         for qa_info in paper_data["qas"]:
        def get_answers_and_evidence(intertext_docs, text_evidence_only):
            answers_and_evidence = {}
            for intertext_doc in intertext_docs:
                for qa_info in intertext_doc.meta["qas"]:
                    question_id = qa_info["question_id"]
                    references = []
                    for annotation_info in qa_info["answers"]:
                        answer_info = annotation_info["answer"]
                        if answer_info["unanswerable"]:
                            references.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
                        else:
                            if answer_info["extractive_spans"]:
                                answer = ", ".join(answer_info["extractive_spans"])
                                answer_type = "extractive"
                            elif answer_info["free_form_answer"]:
                                answer = answer_info["free_form_answer"]
                                answer_type = "abstractive"
                            elif answer_info["yes_no"]:
                                answer = "Yes"
                                answer_type = "boolean"
                            elif answer_info["yes_no"] is not None:
                                answer = "No"
                                answer_type = "boolean"
                            else:
                                raise RuntimeError(f"Annotation {answer_info['annotation_id']} does not contain an answer")
                            if text_evidence_only:
                                evidence = [text for text in answer_info["evidence"] if "FLOAT SELECTED" not in text]
                            else:
                                evidence = answer_info["evidence"]
                            references.append({"answer": answer, "evidence": evidence, "type": answer_type})
                    answers_and_evidence[question_id] = references

            return answers_and_evidence

        def evaluate(gold, predicted):
            max_answer_f1s = []
            max_evidence_f1s = []
            max_answer_f1s_by_type = {
                "extractive": [],
                "abstractive": [],
                "boolean": [],
                "none": [],
            }
            num_missing_predictions = 0
            for question_id, references in gold.items():
                if question_id not in predicted:
                    # the following line was added for debugging purposes
                    logger.error(f"Failed to predict question '{question_id}'!")
                    if self.config['task']['count_missing_predictions']:
                        num_missing_predictions += 1
                        max_answer_f1s.append(0.0)
                        max_evidence_f1s.append(0.0)
                    continue
                answer_f1s_and_types = [
                    (token_f1_score(predicted[question_id]["answer"], reference["answer"]),
                     reference["type"])
                    for reference in gold[question_id]
                ]
                max_answer_f1, answer_type = sorted(answer_f1s_and_types, key=lambda x: x[0], reverse=True)[0]
                max_answer_f1s.append(max_answer_f1)
                max_answer_f1s_by_type[answer_type].append(max_answer_f1)
                evidence_f1s = [
                    paragraph_f1_score(predicted[question_id]["evidence"], reference["evidence"])
                    for reference in gold[question_id]
                ]
                max_evidence_f1s.append(max(evidence_f1s))

            mean = lambda x: sum(x) / len(x) if x else 0.0
            return {
                "Answer F1": mean(max_answer_f1s),
                "Answer F1 by type": {key: mean(value) for key, value in max_answer_f1s_by_type.items()},
                "Evidence F1": mean(max_evidence_f1s),
                "Missing predictions": num_missing_predictions
            }

        # transform the predictions in a QASPER-compatible form
        predicted_answers_and_evidence: Dict[str, Any] = {}
        for prediction in predictions:
            predicted_answers_and_evidence[prediction.question_id] = {
                "answer": prediction.answer_text,
                "evidence": [node.content for node in prediction.evidence_nodes]
            }

        # load the gold answers
        if partition == Partition.DEV:
            documents = self.dev_documents
        elif partition == Partition.TEST:
            documents = self.test_documents
        else:
            raise KeyError(f"Unknown partition {partition}!")
        gold_answers_and_evidence = get_answers_and_evidence(documents, self.config["task"]["text_evidence_only"])

        evaluation_output = evaluate(gold_answers_and_evidence, predicted_answers_and_evidence)

        # we additionally compute Rouge scores
        scorer = RougeScorer(["rouge1", "rouge2", "rougeL"])

        def mean(x):
            return sum(x) / len(x)

        multi_mean_rouge_1_scores = []
        multi_mean_rouge_2_scores = []
        multi_mean_rouge_l_scores = []

        multi_max_rouge_1_scores = []
        multi_max_rouge_2_scores = []
        multi_max_rouge_l_scores = []

        for instance, prediction in zip(instances, predictions):
            rouge_1_scores_here = []
            rouge_2_scores_here = []
            rouge_l_scores_here = []

            for ground_truth_answer_text in instance.answer_texts:
                scores = scorer.score(ground_truth_answer_text, prediction.answer_text)
                rouge_1_scores_here.append(scores["rouge1"].fmeasure)
                rouge_2_scores_here.append(scores["rouge2"].fmeasure)
                rouge_l_scores_here.append(scores["rougeL"].fmeasure)

            multi_mean_rouge_1_scores.append(mean(rouge_1_scores_here))
            multi_mean_rouge_2_scores.append(mean(rouge_2_scores_here))
            multi_mean_rouge_l_scores.append(mean(rouge_l_scores_here))

            multi_max_rouge_1_scores.append(max(rouge_1_scores_here))
            multi_max_rouge_2_scores.append(max(rouge_2_scores_here))
            multi_max_rouge_l_scores.append(max(rouge_l_scores_here))

        samples = []
        for instance, prediction in zip(instances[:10], predictions[:10]):
            samples.append((instance.answer_texts, prediction.answer_text))

        # TODO: maybe filter out cases in which the evidence ground truth has errors (e.g., no nodes found)
        return QASPERResult(
            answer_f1=evaluation_output["Answer F1"],
            answer_f1_by_type=evaluation_output["Answer F1 by type"],
            evidence_f1=evaluation_output["Evidence F1"],
            num_missing_predictions=evaluation_output["Missing predictions"],
            multi_mean_rouge_1=mean(multi_mean_rouge_1_scores),
            multi_mean_rouge_2=mean(multi_mean_rouge_2_scores),
            multi_mean_rouge_l=mean(multi_mean_rouge_l_scores),
            multi_max_rouge_1=mean(multi_max_rouge_1_scores),
            multi_max_rouge_2=mean(multi_max_rouge_2_scores),
            multi_max_rouge_l=mean(multi_max_rouge_l_scores),
            samples=samples
        )

    @property
    def train_instances(self) -> List[QASPERInstance]:
        return self._train_instances

    @property
    def dev_instances(self) -> List[QASPERInstance]:
        return self._dev_instances

    @property
    def test_instances(self) -> List[QASPERInstance]:
        return self._test_instances

    def _find_evidence_nodes(self, document: IntertextDocument, answer: Dict[str, Any]) -> List[Node]:
        evidence_nodes = []
        text_evidence_only = self.config["task"]["text_evidence_only"]
        for node in document.nodes:
            if text_evidence_only and "FLOAT SELECTED" in node.content:
                continue
            for evidence_for in node.meta["is_evidence_for"]:
                if evidence_for["annotation_id"] == answer["annotation_id"]:
                    evidence_nodes.append(node)
                    break
        return evidence_nodes

    def _get_answer_text(self, answer: Dict[str, Any]) -> str:
        # this code is adapted from QASPER's get_answers_and_evidence method
        if answer["answer"]["unanswerable"]:
            return "Unanswerable"
        elif answer["answer"]["extractive_spans"]:
            return ", ".join(answer["answer"]["extractive_spans"])
        elif answer["answer"]["free_form_answer"]:
            return answer["answer"]["free_form_answer"]
        elif answer["answer"]["yes_no"]:
            return "Yes"
        elif answer["answer"]["yes_no"] is not None:
            return "No"
        else:
            raise RuntimeError(f"Annotation {answer['answer']['annotation_id']} does not contain an answer")

    def _create_train_instances_from_document(self, doc: IntertextDocument) -> List[QASPERInstance]:
        instances = []

        # create multiple instances per question (one for each answer)
        for qas in doc.meta["qas"]:
            for answer in qas["answers"]:
                answer_text = self._get_answer_text(answer)

                evidence_nodes = [self._find_evidence_nodes(doc, answer)]
                instance = QASPERInstance(
                    question_id=qas["question_id"],
                    document=doc,
                    question=qas["question"],
                    answer_texts=[answer_text],
                    evidence_nodes=evidence_nodes
                )
                instances.append(instance)

        return instances

    def _create_eval_instances_from_document(self, doc: IntertextDocument) -> List[QASPERInstance]:
        instances = []

        # create one instance per question (with all answers)
        for qas in doc.meta["qas"]:
            answer_texts = []
            for answer in qas["answers"]:
                answer_texts.append(self._get_answer_text(answer))

            evidence_nodes = [self._find_evidence_nodes(doc, a) for a in qas["answers"]]
            instance = QASPERInstance(
                question_id=qas["question_id"],
                document=doc,
                question=qas["question"],
                answer_texts=answer_texts,
                evidence_nodes=evidence_nodes
            )
            instances.append(instance)

        return instances

    @staticmethod
    def load_predictions(
            input_path: Path,
            instances: List[QASPERInstance],
    ) -> List[QASPERPrediction]:
        predictions = []
        instance_map = {instance.question_id: instance for instance in instances}
        with open(input_path, "r") as f:
            for line in f:
                prediction = json.loads(line)
                instance = instance_map[prediction["question_id"]]
                evidence_nodes = [
                    instance.document.get_node_by_ix(node_id)
                    for node_id in prediction["evidence_nodes"]
                ]
                predictions.append(QASPERPrediction(
                    question_id=prediction["question_id"],
                    answer_text=prediction["answer_text"],
                    evidence_nodes=evidence_nodes
                ))
        # sort predictions according to the order of the instances
        predictions.sort(key=lambda p: instances.index(instance_map[p.question_id]))
        return predictions