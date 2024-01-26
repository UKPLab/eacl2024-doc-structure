import json
import re
from pathlib import Path
from typing import Dict, List
import os

import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

TASK_SPECIFIC_RESULT_KEYS = {
    'QASPER': [
        'answer_f1',
        'evidence_f1'
    ],
    'Evidence Inference': [
        'classification_macro_f1_score',
        'evidence_detection_f1_score',
    ],
}

METRIC_PRETTY_NAMES = {
    'evidence_detection_f1_score': 'Evi',
    'classification_macro_f1_score': 'Cla',
    'answer_f1': 'Ans',
    'evidence_f1': 'Evi',
    'accuracy': 'Accuracy',
    'r2': 'R2'
}

PROBING_METRICS = {
    'node_type': 'accuracy',
    'sibling': 'accuracy',
    'ancestor': 'accuracy',
    'position': 'accuracy',
    'parent_predecessor': 'accuracy',
    'tree_depth': 'accuracy',
    'structural': 'accuracy'
}

PROBE_PRETTY_NAMES = {
    'node_type': 'Nod',
    'sibling': 'Sib',
    'ancestor': 'Anc',
    'position': 'Pos',
    'parent_predecessor': 'Par',
    'tree_depth': 'Tre',
    'structural': 'Str'
}

MODEL_PRETTY_NAMES = {
    'led-base-16384': 'LED',
    'long-t5-tglobal-base': 'LongT5'
}

TASK_PRETTY_NAMES = {
    'QASPER': 'QAS',
    'Evidence Inference': 'EvI'
}

TEXT_FONTSIZE = 20

def read_data(
        end_task_results_path: Path,
        probing_results_path: Path,
        struct_only: bool = True
):
    # Define files with hashes
    task_hash_mapping_paths = {
        'led-base-16384': {
            'QASPER': '../data/hash_description_mapping_qasper.csv',
            'Evidence Inference': '../data/hash_description_mapping_evidence_inference.csv'
        },
        'long-t5-tglobal-base': {
            'QASPER': '../data/hash_description_mapping_longt5_qasper.csv',
            'Evidence Inference': '../data/hash_description_mapping_longt5_evidence_inference.csv'
        }
    }
    # Get all files in end task results path
    end_task_results_filenames = [
        str(filename)
        for filename in os.listdir(end_task_results_path)
    ]

    results = {}

    for model_name in task_hash_mapping_paths.keys():
        results[model_name] = {}
        for task_name, task_hash_mapping_path in task_hash_mapping_paths[model_name].items():
            # Read hashes
            hash_mapping = pd.read_csv(task_hash_mapping_path)
            # Store results in dict
            results[model_name][task_name] = {
                'model': [],
                'f1000_probing_dirname': []
            }
            # Add keys for task specific metrics
            results[model_name][task_name].update({
                METRIC_PRETTY_NAMES[metric_name]: []
                for metric_name in TASK_SPECIFIC_RESULT_KEYS[task_name]
            })
            # Add keys for probing metrics
            results[model_name][task_name].update({
                PROBE_PRETTY_NAMES[probe_name]: []
                for probe_name in PROBING_METRICS.keys()
            })
            # Iterate over hashes
            for probing_dirname, test_hash in zip(
                    hash_mapping['F1000 Probing Dirname'],
                    hash_mapping['Test Hash']
            ):
                probing_dirname = probing_dirname.strip()
                regex = f'{task_name}.*{test_hash}.json'
                # Get end task results
                for filename in end_task_results_filenames:
                    if re.match(regex, filename):
                        with open(end_task_results_path/filename) as f:
                            end_task_results = json.load(f)
                        break

                results[model_name][task_name]['f1000_probing_dirname'].append(probing_dirname)

                # Get description of configuration
                results[model_name][task_name]['model'].append(
                    hash_mapping[
                        hash_mapping['Test Hash'] == test_hash
                    ]['Description'].values[0])

                for metric_name in TASK_SPECIFIC_RESULT_KEYS[task_name]:
                    results[model_name][task_name][METRIC_PRETTY_NAMES[metric_name]].append(
                        end_task_results['test_result'][metric_name])

                # Get probing results
                for probe_name in PROBING_METRICS.keys():
                    try:
                        with open(probing_results_path / model_name / probing_dirname / probe_name / 'probing.json') as f:
                            probe_results = json.load(f)

                        if PROBING_METRICS[probe_name] == 'compression':
                            results[model_name][task_name][PROBE_PRETTY_NAMES[probe_name]].append(probe_results['minimum_description_length'][PROBING_METRICS[probe_name]])
                        else:
                            results[model_name][task_name][PROBE_PRETTY_NAMES[probe_name]].append(probe_results['metrics'][PROBING_METRICS[probe_name]])

                    except FileNotFoundError:
                        results[model_name][task_name][PROBE_PRETTY_NAMES[probe_name]].append(0)

            results[model_name][task_name] = pd.DataFrame(results[model_name][task_name])

    return results

def correlate_fixed_end_task_to_probing(
        data: pd.DataFrame,
        task_name: str,
        handle_nans: str = 'remove_rows',
        probing_metric_suffix: str = ''
) -> Dict[str, Dict[str, float]]:
    """

    :param data: dataframe with data for a single end task
    :param handle_nans: how to handle nans.
    Options: 'remove_rows': Remove rows with nans, calculate correlation for
    the rest
    'skip': Do not compute correlation for the combination of probing task
    and end task metric
    :return:
    """

    # Average end task scores for all runs with same probing hash
    # data = data.groupby('f1000_probing_hash').mean()

    return_dict = {}

    for end_task_metric in TASK_SPECIFIC_RESULT_KEYS[task_name]:
        return_dict[METRIC_PRETTY_NAMES[end_task_metric]] = {}
        end_task_column_name = METRIC_PRETTY_NAMES[end_task_metric]
        for probe_name in PROBE_PRETTY_NAMES.values():
            probing_column_name = probe_name
            end_task_data = data[end_task_column_name]
            probe_data = data[probing_column_name]
            # Check that no entry is NaN
            if (
                (all(~end_task_data.isna()))
                and (all(~probe_data.isna()))
            ):
                correlation = stats.pearsonr(end_task_data, probe_data)
                return_dict[METRIC_PRETTY_NAMES[end_task_metric]][probe_name] = correlation
            elif handle_nans == 'remove_rows':
                filtered_data = data.loc[
                    (~data[end_task_column_name].isna())
                    & (~data[probing_column_name].isna())
                ]
                if len(filtered_data) > 1:
                    filtered_end_task_data = filtered_data[end_task_column_name]
                    filtered_probing_data = filtered_data[probing_column_name]
                    correlation = stats.pearsonr(filtered_end_task_data, filtered_probing_data)
                    return_dict[end_task_metric][probe_name] = correlation

    return return_dict


def plot_correlation(
        correlations: Dict[str, Dict[str, Dict[str, List[int]]]],
        out_path: Path,
        individual_plots: bool = False
):
    sns.set_context('paper')
    n_models = len(correlations)
    n_end_tasks = len(correlations[list(correlations.keys())[0]])
    if not individual_plots:
        cm = 1 / 2.54
        fig, axes = plt.subplots(
            1,
            n_models * n_end_tasks,
            figsize=(8 * cm, 5 * cm),
            gridspec_kw={'width_ratios':[1, 1, 1, 1.3]}
        )
    else:
        axes = [None for _ in range(n_models * n_end_tasks)]

    titles = []
    for i, model_name in enumerate(correlations):
        for j, end_task in enumerate(correlations[model_name]):
            titles.append(end_task)
            plot_id = i * n_models + j
            heatmap_data = {}
            annot_data = {}
            for end_task_metric in correlations[model_name][end_task]:
                index = [
                    k for k in correlations[model_name][end_task][end_task_metric].keys()
                ]
                heatmap_data[end_task_metric] = []
                annot_data[end_task_metric] = []
                for probing_task in index:
                    r, p = correlations[model_name][end_task][end_task_metric][probing_task]
                    heatmap_data[end_task_metric].append(r)
                    if p < 0.05:
                        annot_data[end_task_metric].append(f'{r:.2f}*')
                    else:
                        annot_data[end_task_metric].append(f'{r:.2f}')
            heatmap_data = pd.DataFrame(
                index=index,
                data=heatmap_data
            )
            annot_data = pd.DataFrame(
                index=index,
                data=annot_data
            )

            add_colorbar = True
            if not individual_plots:
                # Add colorbar only to upper right plot
                if i + 1 != n_models:
                    add_colorbar = False
                if j + 1 != n_end_tasks:
                    add_colorbar = False


            heatmap = sns.heatmap(
                data=heatmap_data,
                annot=annot_data,
                fmt='s',
                center=0,
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                vmax=1,
                vmin=-1,
                ax=axes[plot_id],
                cbar=add_colorbar,
                annot_kws={"fontsize": 6}
            )

            if individual_plots:
                plt.title(end_task)
                plt.tight_layout()
                plt.savefig(out_path / f'correlation_{end_task}_f1000-probing.png', dpi=300)
                plt.clf()

    if not individual_plots:
        for i, ax in enumerate(axes):
            ax.set_title(titles[i])
            ax.tick_params(length=0)
            if i > 0:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            if ax.collections[0].colorbar is not None:
                ax.collections[0].colorbar.set_ticks([-1, 0, 1], length=0)
                ax.collections[0].colorbar.ax.tick_params(length=0, labelsize=6)
                for t in ax.collections[0].colorbar.ax.get_yticklines():
                    t.set_color('white')
            ax.set_xticklabels(
                labels=ax.get_xticklabels(),
                rotation=0
            )
            plt.suptitle(
                '    LED                   LongT5',
                fontsize=10
            )
        plt.tight_layout(rect=[0, -0.05, 1, 1.10])
        plt.savefig(out_path / 'correlation_f1000-probing.pdf')
        plt.clf()

def main(
        end_task_results_path: Path,
        probing_results_path: Path,
        out_path: Path,
        struct_only: bool = True,
        individual_plots: bool = False
):

    data = read_data(
        end_task_results_path=end_task_results_path,
        probing_results_path=probing_results_path,
        struct_only=struct_only
    )
    correlations = {}
    for model_name, model_data in data.items():
        correlations[MODEL_PRETTY_NAMES[model_name]] = {}
        for task_name, df in model_data.items():
            correlations[MODEL_PRETTY_NAMES[model_name]][TASK_PRETTY_NAMES[task_name]] = correlate_fixed_end_task_to_probing(
                data=data[model_name][task_name],
                task_name=task_name
            )
    plot_correlation(
        correlations,
        out_path,
        individual_plots=individual_plots
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--end_task_results_path',
        type=Path,
    )
    parser.add_argument(
        '--probing_results_path',
        type=Path,
    )
    parser.add_argument(
        '--out_path',
        type=Path,
    )
    parser.add_argument(
        '--include_no_struct',
        action='store_true'
    )
    parser.add_argument(
        '--individual_plots',
        action='store_true'
    )
    parser.set_defaults(
        end_task_results_path=Path('../data/results'),
        probing_results_path=Path('../data/probing_results/in_n_out/f1000rd-full'),
        out_path=Path('../data/plots'),
        include_no_struct=False,
        individual_plots=False
    )
    args = parser.parse_args()
    main(
        end_task_results_path=args.end_task_results_path,
        probing_results_path=args.probing_results_path,
        out_path=args.out_path,
        struct_only=not args.include_no_struct,
        individual_plots=args.individual_plots
    )