from typing import Dict, List, Any
from functools import reduce
from pathlib import Path
import json

import pandas as pd
from intertext_graph.itgraph import IntertextDocument, Etype
import matplotlib.pyplot as plt

def apply_filters(
        df: pd.DataFrame,
        filters: Dict[str, List[Any]]
):
    """Helper function that applies a sequence of .isin filters to a
    dataframe
    :param df: dataframe to filter
    :param filters: A dictionary of filters, with the column names as keys
    and lists of allowed values as dict values"""
    assert len(filters) > 0
    comparisons = [
        df[k].isin(v) for k, v in filters.items()
    ]
    return reduce(
        lambda c1, c2: c1 & c2, comparisons[1:],
        comparisons[0]
    )


def normalize(
        df: pd.DataFrame,
        column_to_be_normalized: str,
        normalize_condition_column: str
) -> pd.Series:
    max_values = df.groupby(normalize_condition_column).max()
    min_values = df.groupby(normalize_condition_column).min()


def analyze_evidence_distribution(
        data: List[IntertextDocument],
        count_multiple_annotations: bool = True
) -> Dict[str, List[Any]]:
    def _get_relative_idx_distribution(
            n_nodes_: List[int],
            all_nodes_with_evidence_idx_: List[List[int]]
    ) -> List[float]:
        relative_idxs = []
        for k, (n_nodes_in_itg, sublist) in enumerate(zip(n_nodes_, all_nodes_with_evidence_idx_)):
            for idx in sublist:
                if idx >= 0:
                    relative_idxs.append(idx / n_nodes_in_itg)

        return relative_idxs

    def _flatten(l: List[List[Any]]) -> List[Any]:
        return [
            item
            for sublist in l
            for item in sublist
        ]

    n_nodes = []
    all_node_depths = []
    all_node_types = []
    all_nodes_with_evidence_idx = []
    all_nodes_with_evidence_type = []
    all_nodes_with_evidence_depth = []

    n_no_evidence = 0

    for itg in data:
        node_depths = []
        node_types = []
        nodes_with_evidence_idx = []
        nodes_with_evidence_type = []
        nodes_with_evidence_depth = []



        for i, node in enumerate(itg.unroll_graph()):
            n_evidences = len(node.meta['is_evidence_for'])
            node_type = node.ntype
            node_depth = len(list(itg.breadcrumbs(node, Etype.PARENT)))
            node_depths.append(node_depth)
            node_types.append(node_type)
            if n_evidences > 0:
                node_type = node.ntype
                if count_multiple_annotations:
                    nodes_with_evidence_idx.extend([i]*n_evidences)
                    nodes_with_evidence_type.extend([node_type]*n_evidences)
                    nodes_with_evidence_depth.extend([node_depth]*n_evidences)
                else:
                    nodes_with_evidence_idx.append(i)
                    nodes_with_evidence_type.append(node_type)
                    nodes_with_evidence_depth.append(node_depth)

        if all(
            len(l) > 0
            for l in [
                nodes_with_evidence_idx,
                nodes_with_evidence_type,
                nodes_with_evidence_depth
            ]
        ):
            all_nodes_with_evidence_idx.append(nodes_with_evidence_idx)
            all_nodes_with_evidence_type.append(nodes_with_evidence_type)
            all_nodes_with_evidence_depth.append(nodes_with_evidence_depth)
            n_nodes.append(len(itg.nodes))

        else:
            all_nodes_with_evidence_idx.append([-1])
            all_nodes_with_evidence_type.append(['No evidence'])
            all_nodes_with_evidence_depth.append([-1])
            n_nodes.append(len(itg.nodes))
            n_no_evidence += 1

        all_node_types.append(node_types)
        all_node_depths.append(node_depths)

    print(f'{n_no_evidence} documents with no evidence.')

    all_nodes_with_relative_idx = _get_relative_idx_distribution(
        n_nodes,
        all_nodes_with_evidence_idx
    )

    return {
        'n_nodes': n_nodes,
        'node_depth': _flatten(all_node_depths),
        'node_type': _flatten(all_node_types),
        'evidence_node_idx': _flatten(all_nodes_with_evidence_idx),
        'evidence_relative_node_idx': all_nodes_with_relative_idx,
        'evidence_node_depth': _flatten(all_nodes_with_evidence_depth),
        'evidence_node_type': _flatten(all_nodes_with_evidence_type)
    }


def plot_evidence_distribution(
        path_to_datasets: Path,
        dataset_name: str,
        subset_names: List[str],
        count_multiple_annotations: bool = True
):
    data = []
    for subset_name in subset_names:
        with open(path_to_datasets / dataset_name / subset_name, 'r') as file:
            for line in file.readlines():
                data.append(IntertextDocument._from_json(
                    json.loads(line)
                ))

    distributions = analyze_evidence_distribution(
        data,
        count_multiple_annotations=count_multiple_annotations
    )
    f, axs = plt.subplots(3, 3)
    subplot_map = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2)
    ]
    f.tight_layout()
    # plt.subplots_adjust(top=2)
    for i, (distribution_name, distribution_data) in enumerate(distributions.items()):
        if type(distribution_data[0]) is str:
            data_series = pd.Series(distribution_data)
            counts = data_series.value_counts()
            counts.plot.bar(
                title=distribution_name,
                ax=axs[subplot_map[i][0], subplot_map[i][1]]
            )
        else:
            data_series = pd.Series(distribution_data)
            data_series.plot.hist(
                title=distribution_name,
                ax=axs[subplot_map[i][0], subplot_map[i][1]]
            )

