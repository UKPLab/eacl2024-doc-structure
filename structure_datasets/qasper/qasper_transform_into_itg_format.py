"""
Transform the QASPER dataset into the ITG format.
"""
import json
import logging
import os
from collections import deque
from typing import Any, Dict, Deque

import tqdm
from intertext_graph.itgraph import IntertextDocument, Node, Edge, Etype
from intertext_graph.parsers.itparser import IntertextParser

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

NTYPE_PARAGRAPH = "p"
NTYPE_TITLE = "title"
NTYPE_ARTICLE_TITLE = "article-title"
NTYPE_ABSTRACT = "abstract"

NTYPE_FIGURES_AND_TABLES_SECTION = "figures-and-tables-section"
NTYPE_FIGURE_OR_TABLE = "figure-or-table"


class QASPERParser(IntertextParser):
    """
    Parser to transform a QASPER document into an IntertextDocument.

    Description of the QASPER dataset: https://allenai.org/project/qasper/home
    """

    def __init__(self, dataset_path: str, arxiv_id: str, deep_or_shallow: str):
        """
        Initialize the QASPERParser for a particular paper.

        :param dataset_path: path of the JSON file that contains the document
        :param arxiv_id: arXiv identifier of the particular paper
        """
        super(QASPERParser, self).__init__(dataset_path)
        self._dataset_path: str = dataset_path
        self._arxiv_id: str = arxiv_id
        self.deep_or_shallow = deep_or_shallow

    def __call__(self) -> IntertextDocument:
        """
        Parse the QASPER document into an IntertextDocument.

        :return: the IntertextDocument
        """
        # load the file contents
        with open(self._dataset_path, "r", encoding="utf-8") as file:
            dataset_json = json.load(file)

        # find correct document based on the arXiv identifier
        if self._arxiv_id in dataset_json.keys():
            document_json = dataset_json[self._arxiv_id]
        else:
            logger.error(f"Could not find the document with the arXiv identifier {self._arxiv_id}!")
            assert False, f"Could not find the document with the arXiv identifier {self._arxiv_id}!"

        return self._parse_document(document_json)

    @classmethod
    def _batch_func(cls, path: Any) -> Any:
        raise NotImplementedError  # TODO: implement this

    def _parse_document(self, document_json: Dict[str, Any]) -> IntertextDocument:
        """
        Parse the given QASPER Document.

        :param document_json: QASPER representation of the document
        :return: resulting IntertextDocument
        """

        # create intertext document
        prefix = self._arxiv_id
        metadata = {
            "arxiv_id": self._arxiv_id,
            "qas": document_json["qas"]
        }

        intertext_document = IntertextDocument(
            nodes=[],
            edges=[],
            prefix=prefix,
            meta=metadata
        )

        # create article title as root
        article_title_node = Node(
            content=document_json["title"],
            ntype=NTYPE_ARTICLE_TITLE,
            meta=self._get_evidence_info_as_meta(document_json["title"], document_json["qas"])
        )
        intertext_document.add_node(article_title_node)
        pred_node = article_title_node

        # parse abstract
        abstract_title_node = Node(
            content="Abstract",  # magic string
            ntype=NTYPE_ABSTRACT,
            meta={
                "is_evidence_for": []
            }
        )
        intertext_document.add_node(abstract_title_node)

        abstract_section_parent_edge = Edge(
            src_node=article_title_node,
            tgt_node=abstract_title_node,
            etype=Etype.PARENT
        )
        intertext_document.add_edge(abstract_section_parent_edge)

        abstract_section_next_edge = Edge(
            src_node=pred_node,
            tgt_node=abstract_title_node,
            etype=Etype.NEXT
        )
        intertext_document.add_edge(abstract_section_next_edge)
        pred_node = abstract_title_node

        abstract_content_node = Node(
            content=document_json["abstract"],
            ntype=NTYPE_PARAGRAPH,
            meta=self._get_evidence_info_as_meta(document_json["abstract"], document_json["qas"])
        )
        intertext_document.add_node(abstract_content_node)

        abstract_content_parent_edge = Edge(
            src_node=abstract_title_node,
            tgt_node=abstract_content_node,
            etype=Etype.PARENT
        )
        intertext_document.add_edge(abstract_content_parent_edge)

        abstract_content_next_edge = Edge(
            src_node=pred_node,
            tgt_node=abstract_content_node,
            etype=Etype.NEXT
        )
        intertext_document.add_edge(abstract_content_next_edge)
        pred_node = abstract_content_node

        section_node_stack: Deque[Node] = deque()

        # parse body text
        for current_section_json in document_json["full_text"]:
            if current_section_json["section_name"] is not None:
                current_section_title_content = current_section_json["section_name"]
            else:
                current_section_title_content = ""  # TODO: handle this better?
                logger.warning("The section title is None! ==> use empty string instead")

            parent_section_node: Node = None

            if self.deep_or_shallow == 'deep':

                current_section_title_content_split = current_section_title_content.split(' ::: ')
                current_section_title_content = current_section_title_content_split[-1]

                if len(current_section_title_content_split) > 1:
                    parent_section_name = current_section_title_content_split[-2]
                    while section_node_stack and not parent_section_node:
                        candidate_parent_section_node = section_node_stack[-1]
                        if parent_section_name == candidate_parent_section_node.content:
                            parent_section_node = candidate_parent_section_node
                        else:
                            section_node_stack.pop()

            if parent_section_node is None:
                parent_section_node = article_title_node

            # add the section title
            current_section_title_node = Node(
                content=current_section_title_content,
                ntype=NTYPE_TITLE,
                meta=self._get_evidence_info_as_meta(current_section_title_content, document_json["qas"])
            )
            intertext_document.add_node(current_section_title_node)
            section_node_stack.append(current_section_title_node)

            current_section_title_parent_edge = Edge(
                src_node=parent_section_node,
                tgt_node=current_section_title_node,
                etype=Etype.PARENT
            )
            intertext_document.add_edge(current_section_title_parent_edge)

            current_section_title_next_edge = Edge(
                src_node=pred_node,
                tgt_node=current_section_title_node,
                etype=Etype.NEXT
            )
            intertext_document.add_edge(current_section_title_next_edge)
            pred_node = current_section_title_node

            # parse the paragraphs
            for paragraph_text in current_section_json["paragraphs"]:
                current_paragraph_node = Node(
                    content=paragraph_text,
                    ntype=NTYPE_PARAGRAPH,
                    meta=self._get_evidence_info_as_meta(paragraph_text, document_json["qas"])
                )
                intertext_document.add_node(current_paragraph_node)

                current_paragraph_parent_edge = Edge(
                    src_node=current_section_title_node,
                    tgt_node=current_paragraph_node,
                    etype=Etype.PARENT
                )
                intertext_document.add_edge(current_paragraph_parent_edge)

                current_paragraph_next_edge = Edge(
                    src_node=pred_node,
                    tgt_node=current_paragraph_node,
                    etype=Etype.NEXT
                )
                intertext_document.add_edge(current_paragraph_next_edge)
                pred_node = current_paragraph_node

        # parse figures and tables
        figures_and_tables_title_node = Node(
            content="Figures and Tables",  # magic string
            ntype=NTYPE_FIGURES_AND_TABLES_SECTION,
            meta={
                "is_evidence_for": []
            }
        )
        intertext_document.add_node(figures_and_tables_title_node)

        figures_and_tables_title_parent_edge = Edge(
            src_node=article_title_node,
            tgt_node=figures_and_tables_title_node,
            etype=Etype.PARENT
        )
        intertext_document.add_edge(figures_and_tables_title_parent_edge)

        figures_and_tables_title_next_edge = Edge(
            src_node=pred_node,
            tgt_node=figures_and_tables_title_node,
            etype=Etype.NEXT
        )
        intertext_document.add_edge(figures_and_tables_title_next_edge)
        pred_node = figures_and_tables_title_node

        for figure_or_table_json in document_json["figures_and_tables"]:
            meta = self._get_evidence_info_as_meta(figure_or_table_json["caption"], document_json["qas"])
            meta["file"] = figure_or_table_json["file"]
            current_figure_or_table_node = Node(
                content=figure_or_table_json["caption"],
                ntype=NTYPE_FIGURE_OR_TABLE,
                meta=meta
            )
            intertext_document.add_node(current_figure_or_table_node)

            current_figure_or_table_parent_edge = Edge(
                src_node=figures_and_tables_title_node,
                tgt_node=current_figure_or_table_node,
                etype=Etype.PARENT
            )
            intertext_document.add_edge(current_figure_or_table_parent_edge)

            current_figure_or_table_next_edge = Edge(
                src_node=pred_node,
                tgt_node=current_figure_or_table_node,
                etype=Etype.NEXT
            )
            intertext_document.add_edge(current_figure_or_table_next_edge)
            pred_node = current_figure_or_table_node

        return intertext_document

    def _get_evidence_info_as_meta(self, text, qas):
        meta = {
            "is_evidence_for": []
        }
        for question in qas:
            for answer in question["answers"]:
                for ix, evidence in enumerate(answer["answer"]["evidence"]):
                    if "FLOAT SELECTED: " in evidence:
                        evidence = evidence.replace("FLOAT SELECTED: ", "")
                    if "FLOAT SELECTED" in evidence:
                        evidence = evidence.replace("FLOAT SELECTED", "")
                    text = text.replace("\n", " ")
                    if evidence.lower().strip() == text.lower().strip():
                        meta["is_evidence_for"].append({
                            "annotation_id": answer["annotation_id"],
                            "evidence_ix": ix
                        })
                        break
        return meta


def transform_document(dataset_path: str, arxiv_id: str, deep_or_shallow: str) -> IntertextDocument:
    """
    Transform the QASPER document with the given arxiv id into ITG format.

    :param dataset_path: path of the JSON file that contains the document
    :param arxiv_id: arXiv identifier of the particular paper
    :return: resulting IntertextDocument
    """
    qasper_parser = QASPERParser(dataset_path, arxiv_id, deep_or_shallow)
    intertext_document = qasper_parser()
    return intertext_document


if __name__ == "__main__":
    logger.info("Transform the QASPER dataset into the ITG format.")

    config.load_config_json_file("../path_config_local.json", include_in_hash=False)

    # load the data
    train_path = os.path.join(config.get("path.QASPER"), "qasper-train-v0.3.json")
    dev_path = os.path.join(config.get("path.QASPER"), "qasper-dev-v0.3.json")
    test_path = os.path.join(config.get("path.QASPER"), "qasper-test-v0.3.json")


    with open(train_path, "r", encoding="utf-8") as file:
        train_data = json.load(file)

    with open(dev_path, "r", encoding="utf-8") as file:
        dev_data = json.load(file)

    with open(test_path, "r", encoding="utf-8") as file:
        test_data = json.load(file)

    logger.info(f"Gathered {len(train_data)} train instances, {len(dev_data)} dev instances and "
                f"{len(test_data)} test instances.")

    for conf in ['deep', 'shallow']:
        logger.info(f'Doing {conf} transformation.')

        logger.info("Transform the training instances.")
        train_instances = []
        for arxiv_id in tqdm.tqdm(train_data.keys()):
            train_instances.append(transform_document(train_path, arxiv_id, conf))

        logger.info("Transform the development instances.")
        dev_instances = []
        for arxiv_id in tqdm.tqdm(dev_data.keys()):
            dev_instances.append(transform_document(dev_path, arxiv_id, conf))

        logger.info("Transform the test instances.")
        test_instances = []
        for arxiv_id in tqdm.tqdm(test_data.keys()):
            test_instances.append(transform_document(test_path, arxiv_id, conf))

        train_result_path = os.path.join(config.get("path.QASPER-ITG"), f"{conf}-train.jsonl")
        with open(train_result_path, "w", encoding="utf-8") as file:
            file.write("\n".join(train_instance.to_json(indent=None) for train_instance in train_instances))

        dev_result_path = os.path.join(config.get("path.QASPER-ITG"), f"{conf}-dev.jsonl")
        with open(dev_result_path, "w", encoding="utf-8") as file:
            file.write("\n".join(dev_instance.to_json(indent=None) for dev_instance in dev_instances))

        test_result_path = os.path.join(config.get("path.QASPER-ITG"), f"{conf}-test.jsonl")
        with open(test_result_path, "w", encoding="utf-8") as file:
            file.write("\n".join(test_instance.to_json(indent=None) for test_instance in test_instances))

    logger.info(f"All done with configuration '{config.get_config_hash()}'!")
