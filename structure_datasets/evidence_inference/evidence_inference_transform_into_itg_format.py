"""
Transform the Evidence Inference dataset into the ITG format.
"""
import collections
import csv
import glob
import logging
import os
from typing import Any, Dict, List

import fuzzysearch
import tqdm
from intertext_graph.itgraph import IntertextDocument, Node, Edge, Etype
from intertext_graph.parsers.itparser import IntertextParser

import config
from evidence_inference.article_reader import Article

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

NTYPE_PARAGRAPH = "p"
NTYPE_TITLE = "title"
NTYPE_ARTICLE_TITLE = "article-title"
NTYPE_ABSTRACT = "abstract"

# Although each of these prompts have gone through our 3-stage process, there is still some noise in the dataset.
# Here we have flagged some of the prompt ids that may be a problem. We have split up this into 3 groups:
# - Incorrect: obviously wrong and should not be included.
# - Questionable: requires closer examination of the article to make a full decision on the quality of prompt.
# - Somewhat malformed: do not have the proper context or details needed.
incorrect_prompt_ids = [
    '911', '912', '1262', '1261', '3044', '3248', '3111', '3620', '4308', '4490', '4491', '4324',
    '4325', '4492', '4824', '5000', '5001', '5002', '5046', '5047', '4948', '5639', '5710', '5752',
    '5775', '5782', '5841', '5843', '5861', '5862', '5863', '5964', '5965', '5966', '5975', '4807',
    '5776', '5777', '5778', '5779', '5780', '5781', '6034', '6065', '6066', '6666', '6667', '6668',
    '6669', '7040', '7042', '7944', '8590', '8605', '8606', '8639', '8640', '8745', '8747', '8749',
    '8877', '8878', '8593', '8631', '8635', '8884', '8886', '8773', '10032', '10035', '8876', '8875',
    '8885', '8917', '8921', '8118', '10885', '10886', '10887', '10888', '10889', '10890'
]

questionable_prompt_ids = [
    '7811', '7812', '7813', '7814', '7815', '8197', '8198', '8199', '8200', '8201', '9429', '9430',
    '9431', '8536', '9432'
]

somewhat_malformed_prompt_ids = [
    '3514', '346', '5037', '4715', '8767', '9295', '9297', '8870', '9862'
]

excluded_prompt_ids = incorrect_prompt_ids + questionable_prompt_ids + somewhat_malformed_prompt_ids
excluded_prompt_ids = set(excluded_prompt_ids)


class EvidenceInferenceParser(IntertextParser):
    """
    Parser to transform an Evidence Inference document into an IntertextDocument.

    Description of the PubMed XML format: https://pubmed.ncbi.nlm.nih.gov/help/
    """

    def __init__(self, nxml_path: str, txt_path: str, prompts_path: str, annotations_path: str, mode: str):
        """
        Initialize the EvidenceInferenceParser for a particular paper.

        :param nxml_path: path of the XML file
        :param txt_path: path of the TXT file
        :param prompts_path: path to the prompts CSV file
        :param annotations_path: path to the annotations CSV file
        :param mode: whether to create a 'shallow' or 'deep' ITG document
        """
        super(EvidenceInferenceParser, self).__init__(nxml_path)
        self._nxml_path: str = nxml_path
        self._txt_path: str = txt_path
        self._prompts_path: str = prompts_path
        self._annotations_path: str = annotations_path
        self._mode: str = mode

    def __call__(self) -> IntertextDocument:
        """
        Parse the Evidence Inference document into an IntertextDocument.

        :return: the IntertextDocument
        """
        # parse the XML file with the Evidence Inference parser
        article = Article(xml_path=self._nxml_path)

        # load the TXT file
        with open(self._txt_path, "r", encoding="utf-8") as file:
            article_txt = file.read()

        # load prompts and annotations
        with open(self._prompts_path, "r", encoding="utf-8") as file:
            prompts_list = list(csv.DictReader(file))

        with open(self._annotations_path, "r", encoding="utf-8") as file:
            annotations_list = list(csv.DictReader(file))

        return self._parse_document(article, article_txt, prompts_list, annotations_list)

    def _parse_document(self, article, article_txt, prompts_list, annotations_list) -> IntertextDocument:
        """
        Parse the given Evidence Inference Document.

        :param article: Evidence Inference document parsed by the Evidence Inference article reader
        :param article_txt: Evidence Inference document as loaded from the TXT file
        :param prompts_list: list of Evidence Inference prompts
        :param annotations_list: list of Evidence Inference annotations
        :return: resulting IntertextDocument
        """
        # create intertext document
        prefix = article.get_pmcid()
        prompts = self._transform_prompts(article, article_txt, prompts_list)
        annotations = self._transform_annotations(article, article_txt, annotations_list)
        metadata = {
            "pmc_id": article.get_pmcid(),
            "prompts": prompts,
            "annotations": annotations
        }

        intertext_document = IntertextDocument(
            nodes=[],
            edges=[],
            prefix=prefix,
            meta=metadata
        )

        # create article title as root
        article_title_node = Node(
            content=article.get_title(),
            ntype=NTYPE_ARTICLE_TITLE,
            meta=self._get_evidence_info_as_meta(article.get_title(), annotations)
        )
        intertext_document.add_node(article_title_node)
        pred_node = article_title_node

        if self._mode == "shallow":
            # parse abstract
            for section_title, section_paragraphs in article.article_dict.items():
                if section_title.startswith("abstract"):
                    abstract_title_node = Node(
                        content=".".join(section_title.split(".")[1:]),
                        ntype=NTYPE_ABSTRACT,
                        meta=self._get_evidence_info_as_meta(section_title, annotations)
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

                    for section_paragraph in section_paragraphs:
                        abstract_content_node = Node(
                            content=section_paragraph,
                            ntype=NTYPE_PARAGRAPH,
                            meta=self._get_evidence_info_as_meta(section_paragraph, annotations)
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

            # parse body text
            for section_title, section_paragraphs in article.article_dict.items():
                if section_title.startswith("body"):
                    body_title_node = Node(
                        content=".".join(section_title.split(".")[1:]),
                        ntype=NTYPE_TITLE,
                        meta=self._get_evidence_info_as_meta(section_title, annotations)
                    )
                    intertext_document.add_node(body_title_node)

                    body_section_parent_edge = Edge(
                        src_node=article_title_node,
                        tgt_node=body_title_node,
                        etype=Etype.PARENT
                    )
                    intertext_document.add_edge(body_section_parent_edge)

                    body_section_next_edge = Edge(
                        src_node=pred_node,
                        tgt_node=body_title_node,
                        etype=Etype.NEXT
                    )
                    intertext_document.add_edge(body_section_next_edge)
                    pred_node = body_title_node

                    for section_paragraph in section_paragraphs:
                        body_content_node = Node(
                            content=section_paragraph,
                            ntype=NTYPE_PARAGRAPH,
                            meta=self._get_evidence_info_as_meta(section_paragraph, annotations)
                        )
                        intertext_document.add_node(body_content_node)

                        body_content_parent_edge = Edge(
                            src_node=body_title_node,
                            tgt_node=body_content_node,
                            etype=Etype.PARENT
                        )
                        intertext_document.add_edge(body_content_parent_edge)

                        body_content_next_edge = Edge(
                            src_node=pred_node,
                            tgt_node=body_content_node,
                            etype=Etype.NEXT
                        )
                        intertext_document.add_edge(body_content_next_edge)
                        pred_node = body_content_node
        elif self._mode == "deep":
            stack_title_parts = []
            stack_nodes = []

            for title, paragraphs in article.article_dict.items():
                current_title_parts = title.split(".")

                # determine to what depth this node is similar to the previous node
                diff_depth = 0
                for depth, (stack_title_part, current_title_part) in enumerate(
                        zip(stack_title_parts, current_title_parts)):
                    if stack_title_part != current_title_part:
                        diff_depth = depth
                        break

                # remove the nodes that are not similar from the stack
                stack_nodes = stack_nodes[:diff_depth]
                stack_title_parts = stack_title_parts[:diff_depth]

                # choose the common ancestor
                if diff_depth == 0:
                    common_ancestor = article_title_node
                else:
                    common_ancestor = stack_nodes[-1]

                # add the rest of the current title parts as new section title nodes
                parent_node = common_ancestor
                for current_title_part in current_title_parts[diff_depth:]:
                    # skip 'body' so that the body sections have the article title as parent
                    if current_title_part == "body":
                        continue

                    title_node = Node(
                        content=current_title_part,
                        ntype=NTYPE_ABSTRACT if current_title_part == "abstract" and stack_title_parts == [] else NTYPE_TITLE,
                        meta=self._get_evidence_info_as_meta(current_title_part, annotations)
                    )
                    intertext_document.add_node(title_node)

                    title_parent_edge = Edge(
                        src_node=parent_node,
                        tgt_node=title_node,
                        etype=Etype.PARENT
                    )
                    intertext_document.add_edge(title_parent_edge)
                    parent_node = title_node

                    title_next_edge = Edge(
                        src_node=pred_node,
                        tgt_node=title_node,
                        etype=Etype.NEXT
                    )
                    intertext_document.add_edge(title_next_edge)
                    pred_node = title_node

                    # update the stack
                    stack_title_parts.append(current_title_part)
                    stack_nodes.append(title_node)

                # add the paragraphs
                for paragraph in paragraphs:
                    paragraph_node = Node(
                        content=paragraph,
                        ntype=NTYPE_PARAGRAPH,
                        meta=self._get_evidence_info_as_meta(paragraph, annotations)
                    )
                    intertext_document.add_node(paragraph_node)

                    paragraph_parent_edge = Edge(
                        src_node=parent_node,
                        tgt_node=paragraph_node,
                        etype=Etype.PARENT
                    )
                    intertext_document.add_edge(paragraph_parent_edge)
                    # do not update the parent node since we are not going deeper

                    paragraph_next_edge = Edge(
                        src_node=pred_node,
                        tgt_node=paragraph_node,
                        etype=Etype.NEXT
                    )
                    intertext_document.add_edge(paragraph_next_edge)
                    pred_node = paragraph_node
        else:
            assert False, f"Unknown mode '{self._mode}'!"

        return intertext_document

    def _get_evidence_info_as_meta(self, text, annotations):
        meta = {
            "is_evidence_for": []
        }
        for prompt_id, annotations_here in annotations.items():
            for annotation in annotations_here:
                for ix, evidence_text in enumerate(annotation["evidence_text"]):
                    if evidence_text == "":
                        continue

                    # TODO: token-level evidence
                    matches = fuzzysearch.find_near_matches(evidence_text.lower(), text.lower(), max_l_dist=len(evidence_text) // 20)
                    if matches != []:
                        meta["is_evidence_for"].append({
                            "prompt_id": annotation["prompt_id"],
                            "user_id": annotation["user_id"],
                            "evidence_ix": ix
                        })
        return meta

    def _transform_prompts(self, article, article_txt, prompts_list) -> Dict[str, Dict[str, Any]]:
        transformed_prompts = {}
        for prompt in prompts_list:
            if str(prompt["PMCID"]) == article.get_pmcid():
                assert str(prompt["PromptID"]) not in transformed_prompts.keys()

                if str(prompt["PromptID"]) not in excluded_prompt_ids:
                    transformed_prompts[str(prompt["PromptID"])] = {
                        "prompt_id": str(prompt["PromptID"]),
                        "pmc_id": str(prompt["PMCID"]),
                        "outcome": str(prompt["Outcome"]),
                        "intervention": str(prompt["Intervention"]),
                        "comparator": str(prompt["Comparator"])
                    }
                else:
                    logger.info(f"Excluded prompt {prompt['PromptID']}!")
        return transformed_prompts

    def _transform_annotations(self, article, article_txt, annotations_list) -> Dict[str, List[Dict[str, Any]]]:
        transformed_annotations = collections.defaultdict(dict)
        for annotation in annotations_list:
            if str(annotation["PMCID"]) == article.get_pmcid():

                if annotation["Valid Label"] == "True" and annotation["Valid Reasoning"] == "True":
                    # this is necessary:
                    # Note: Some prompts will have 2 answers by a single doctor. This is because they might cite 2 differen
                    # pieces of evidence. To properly identify this, look for rows with the same 'PromptID' and also the same 'UserID'.
                    if str(annotation["UserID"]) in transformed_annotations[str(annotation["PromptID"])].keys():
                        prev_entry = transformed_annotations[str(annotation["PromptID"])][str(annotation["UserID"])]
                        # make sure that all other values are the same
                        assert str(annotation["PromptID"]) == prev_entry["prompt_id"]
                        assert str(annotation["PMCID"]) == prev_entry["pmc_id"]
                        assert str(annotation["UserID"]) == prev_entry["user_id"]
                        assert (True if annotation["Valid Label"] == "True" else False) == prev_entry["valid_label"]
                        assert (True if annotation["Valid Reasoning"] == "True" else False) == prev_entry["valid_reasoning"]
                        assert str(annotation["Label"]) == prev_entry["label"]
                        assert int(annotation["Label Code"]) == prev_entry["label_code"]

                        # add the new entries
                        prev_entry["annotations"].append(str(annotation["Annotations"]))
                        prev_entry["in_abstract"].append(True if annotation["In Abstract"] == "True" else False)
                        prev_entry["evidence_start"].append(int(annotation["Evidence Start"]))
                        prev_entry["evidence_end"].append(int(annotation["Evidence End"]))
                        prev_entry["evidence_text"].append(article_txt[int(annotation["Evidence Start"]):int(annotation["Evidence End"]) + 1])
                    else:
                        transformed_annotations[str(annotation["PromptID"])][str(annotation["UserID"])] = {
                            "prompt_id": str(annotation["PromptID"]),
                            "pmc_id": str(annotation["PMCID"]),
                            "user_id": str(annotation["UserID"]),
                            "valid_label": True if annotation["Valid Label"] == "True" else False,
                            "valid_reasoning": True if annotation["Valid Reasoning"] == "True" else False,
                            "label": str(annotation["Label"]),
                            "annotations": [str(annotation["Annotations"])],
                            "label_code": int(annotation["Label Code"]),
                            "in_abstract": [True if annotation["In Abstract"] == "True" else False],
                            "evidence_start": [int(annotation["Evidence Start"])],
                            "evidence_end": [int(annotation["Evidence End"])],
                            "evidence_text": [article_txt[int(annotation["Evidence Start"]):int(annotation["Evidence End"]) + 1]]  # end is also inclusive
                        }
                else:
                    logger.info("Excluded invalid annotation!")
        # transform User ID dict into lists
        new_transformed_annotations = {}
        for key, value in transformed_annotations.items():
            new_transformed_annotations[key] = list(value.values())
        return dict(new_transformed_annotations)

    @classmethod
    def _batch_func(cls, path: Any) -> Any:
        raise NotImplementedError  # TODO: implement this


def transform_document(document_id, nxml_file_paths, txt_file_paths, prompts_path, annotations_path) -> IntertextDocument:
    """
    Transform the Evidence Inference document into ITG format.

    :param document_id: id of the document
    :param nxml_file_paths: paths to the Evidence Inference document XML files
    :param txt_file_paths: paths to the Evidence Inference TXT files
    :param prompts_path: path to the prompts CSV file
    :param annotations_path: path to the annotations CSV file
    :return: resulting IntertextDocument
    """
    filtered_nxml_paths = [path for path in nxml_file_paths if f"PMC{document_id}.nxml" in path]
    assert len(filtered_nxml_paths) == 1
    nxml_path = filtered_nxml_paths[0]

    filtered_txt_paths = [path for path in txt_file_paths if f"PMC{document_id}.txt" in path]
    assert len(filtered_txt_paths) == 1
    txt_path = filtered_txt_paths[0]

    evidence_inference_parser = EvidenceInferenceParser(nxml_path, txt_path, prompts_path, annotations_path, config.get("shallow-or-deep"))
    intertext_document = evidence_inference_parser()
    return intertext_document


if __name__ == "__main__":
    logger.info("Transform the Evidence Inference dataset into the ITG format.")

    config.load_config_json_file("../path_config_local.json", include_in_hash=False)
    config.set("shallow-or-deep", "deep")

    # load the data
    nxml_template = os.path.join(config.get("path.EVIDENCE-INFERENCE"), "evidence-inference", "annotations", "xml_files", "*.nxml")
    nxml_file_paths = glob.glob(nxml_template)
    nxml_file_paths.sort()

    txt_template = os.path.join(config.get("path.EVIDENCE-INFERENCE"), "evidence-inference", "annotations", "txt_files", "*.txt")
    txt_file_paths = glob.glob(txt_template)
    txt_file_paths.sort()

    prompts_path = os.path.join(config.get("path.EVIDENCE-INFERENCE"), "evidence-inference", "annotations", "prompts_merged.csv")
    annotations_path = os.path.join(config.get("path.EVIDENCE-INFERENCE"), "evidence-inference", "annotations", "annotations_merged.csv")

    train_ids_path = os.path.join(config.get("path.EVIDENCE-INFERENCE"), "evidence-inference", "annotations", "splits", "train_article_ids.txt")
    dev_ids_path = os.path.join(config.get("path.EVIDENCE-INFERENCE"), "evidence-inference", "annotations", "splits", "validation_article_ids.txt")
    test_ids_path = os.path.join(config.get("path.EVIDENCE-INFERENCE"), "evidence-inference", "annotations", "splits", "test_article_ids.txt")

    logger.info(f"Gathered {len(nxml_file_paths)} XML file paths and {len(txt_file_paths)} TXT file paths.")

    logger.info("Load the document ids.")
    with open(train_ids_path, encoding="utf-8") as file:
        train_ids = [line.strip() for line in file]

    with open(dev_ids_path, encoding="utf-8") as file:
        dev_ids = [line.strip() for line in file]

    with open(test_ids_path, encoding="utf-8") as file:
        test_ids = [line.strip() for line in file]

    logger.info("Transform the documents into ITG format.")

    logger.info("Transform the training instances.")
    train_instances = []
    for train_id in tqdm.tqdm(train_ids):
        train_instances.append(transform_document(train_id, nxml_file_paths, txt_file_paths, prompts_path, annotations_path))

    logger.info("Transform the development instances.")
    dev_instances = []
    for dev_id in tqdm.tqdm(dev_ids):
        dev_instances.append(transform_document(dev_id, nxml_file_paths, txt_file_paths, prompts_path, annotations_path))

    logger.info("Transform the test instances.")
    test_instances = []
    for test_id in tqdm.tqdm(test_ids):
        test_instances.append(transform_document(test_id, nxml_file_paths, txt_file_paths, prompts_path, annotations_path))

    logger.info("Store the results.")
    train_result_path = os.path.join(config.get("path.EVIDENCE-INFERENCE-ITG"), f"{config.get('shallow-or-deep')}-train.jsonl")
    with open(train_result_path, "w", encoding="utf-8") as file:
        file.write("\n".join(instance.to_json(indent=None) for instance in train_instances))

    dev_result_path = os.path.join(config.get("path.EVIDENCE-INFERENCE-ITG"), f"{config.get('shallow-or-deep')}-dev.jsonl")
    with open(dev_result_path, "w", encoding="utf-8") as file:
        file.write("\n".join(instance.to_json(indent=None) for instance in dev_instances))

    test_result_path = os.path.join(config.get("path.EVIDENCE-INFERENCE-ITG"), f"{config.get('shallow-or-deep')}-test.jsonl")
    with open(test_result_path, "w", encoding="utf-8") as file:
        file.write("\n".join(instance.to_json(indent=None) for instance in test_instances))

    logger.info(f"All done with configuration '{config.get_config_hash()}'!")
