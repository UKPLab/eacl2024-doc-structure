#!/usr/bin/env python

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import seaborn as sns


model_map = {
    'led-base-16384/003led': 'LED',
    'led-base-16384/037_F1000_pretrained_LED_vanilla-701c-1388': 'vanilla',
    'led-base-16384/045_F1000_pretrained_LED_is-node-boundaries_probe_struct-f6da-ae8c': 'tok-boundaries',
    'led-base-16384/046_F1000_pretrained_LED_is-node-depths_probe_struct-f56f-5dfe': 'tok-depth',
    'led-base-16384/047_F1000_pretrained_LED_is-node-types_probe_struct-040a-daf9': 'tok-type',
    'led-base-16384/041_F1000_pretrained_LED_pe-node-depths-adb0-6d4a': 'emb-depth',
    'led-base-16384/042_F1000_pretrained_LED_pe-node-types-a9d5-7af4': 'emb-type',
    'led-base-16384/048_F1000_pretrained_LED_pe-node-depths-is-node-types_probe_struct-a73b-bd0b': 'emb-depth-tok-type',
    'led-base-16384/049_F1000_pretrained_LED_pe-node-types-is-node-types_probe_struct-f298-6e7c': 'emb-depth-tok-type',
    'led-base-16384/049_F1000_pretrained_LED_pe-node-types-is-node-depths_probe_struct-9000-7949': 'emb-type-tok-depth',
    'led-base-16384/050_F1000_pretrained_LED_pe-node-depths-is-node-depths_probe_struct-d38b-c0a4': 'emb-depth-tok-depth',
    'long-t5-tglobal-base/031_t5_lr': 'LongT5',
    'long-t5-tglobal-base/031_F1000_pretrained_LongT5_lr-0-1_vanilla-c105-b93e': 'vanilla',
    'long-t5-tglobal-base/034_F1000_pretrained_LongT5_lr-0-1_is-node-boundaries_probe_struct-f68d-d6fd': 'tok-sep',
    'long-t5-tglobal-base/035_F1000_pretrained_LongT5_lr-0-1_is-node-depths_probe_struct-4ae7-8665': 'tok-depth',
    'long-t5-tglobal-base/036_F1000_pretrained_LongT5_lr-0-1_is-node-types_probe_struct-4168-320c': 'tok-type',
    'long-t5-tglobal-base/032_F1000_pretrained_LongT5_lr-0-1_pe-node-depths-f390-28b2': 'emb-depth',
    'long-t5-tglobal-base/033_F1000_pretrained_LongT5_lr-0-1_pe-node-types-403a-cb6a': 'emb-type',
    'long-t5-tglobal-base/037_F1000_pretrained_LongT5_lr-0-1_pe-node-depths-is-node-types_probe_struct-5b77-e1ff': 'emb-depth-tok-type',
    'long-t5-tglobal-base/038_f1000_pretrained_LongT5_lr-0-1_pe-node-types-is-node-types_probe_struct-b21c-43db': 'emb-type-tok-type',
    'long-t5-tglobal-base/039_f1000_pretrained_LongT5_lr-0-1_pe_node_types_is_node_depths_probe_struct_2242-b767': 'emb-type-tok-depth',
    'long-t5-tglobal-base/040_f1000_pretrained_LongT5_lr-0-1_pe_node_depths_is_node_depths_probe_struct_6056-5110': 'emb-depth-tok-depth',
}

probe_map = {
    'ancestor': 'Ancestor',
    'linear_distance': 'Linear Dist',
    'linear_predecessor': 'Linear Pred',
    'node_type': 'Node Type',
    'parent_predecessor': 'Parent Pred',
    'position': 'Position',
    'sibling': 'Sibling',
    'structural': 'Structural',
    'tree_depth': 'Tree Depth'
}


def load_results(
        base: Path,
        models: List[str],
        location: str
) -> Dict[str, List[Dict[str, Any]]]:
    if location == 'cluster':
        raise NotImplementedError

    def open_(path: Path):
        if location == 'local':
            return open(path)
        else:
            raise NotImplementedError

    def check_existence(path: Path, type: str) -> bool:
        if type not in ['f', 'd']:
            raise ValueError
        if location == 'local':
            if type == 'f':
                return path.exists()
            else:
                return path.is_dir()
        else:
            raise NotImplementedError

    def iter_dir(path: Path):
        if location == 'local':
            for filename in path.iterdir():
                yield filename
        else:
            raise NotImplementedError


    results = {}
    for model in models:
        for variant in '', '-atomic', '-random':
            path = base / (model + variant)
            path_exists = check_existence(path, 'd')
            if path_exists:
                name = '/'.join(path.parts[-2:])
                results[name] = []
                for probe in iter_dir(path):
                    if check_existence(probe, 'd') and check_existence(probe / 'probing.json', 'f'):
                        with open_(probe / 'probing.json') as f:
                            results[name].append(json.load(f))

    return results


def probe_results(results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    data = {'model': [], 'probe_name': [], 'online_codelength': [], 'compression': [], 'accuracy': [], 'r2': []}
    for model, probes in results.items():
        model, version = model.split('/')
        if version.endswith('-atomic') or version.endswith('-random'):
            variant = version[-6:]
            version = version[:-7]
            model_name = f"{model_map[f'{model}/{version}']} {'Rand' if variant == 'random' else 'Atom'}"
        else:
            model_name = model_map[f'{model}/{version}']
        for probe in probes:
            data['model'].append(model_name)
            data['probe_name'].append(probe['probe']['probe_name'])
            if 'minimum_description_length' in probe:
                try:
                    data['online_codelength'].append(probe['minimum_description_length']['online_codelength'])
                    data['compression'].append(probe['minimum_description_length']['compression'])
                except KeyError:
                    data['online_codelength'].append(None)
                    data['compression'].append(None)
            else:
                data['online_codelength'].append(None)
                data['compression'].append(None)
            data['accuracy'].append(probe['metrics']['accuracy'] if 'accuracy' in probe['metrics'] else None)
            data['r2'].append(probe['metrics']['r2'] if 'r2' in probe['metrics'] else None)
    return pd.DataFrame(data=data)


def plot_probes(df: pd.DataFrame, metric: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    title = {
        'online_codelength': 'Minimum Description Length',
        'compression': 'Minimum Description Length',
        'accuracy': 'Categorical',
        'r2': 'Regression'}
    df = df.dropna(subset=[metric])
    ax = sns.barplot(x='probe_name', y=metric, hue='model', data=df, order=sorted(df['probe_name'].unique()), ci=None)
    if metric == 'compression':
        ax.set_yscale('log')
    split = out.stem.split('_')[1:]
    ax.set(
        xlabel=None,
        ylabel=' '.join(metric.split('_')).title(),
        title=f"{title[metric]} — {'F1000Research' if split[0].startswith('f1000') else 'Wikipedia'}")
    ax.set_ylim(bottom=-0.1 if len(df[df[metric] < 0]) > 0 else 0)
    ax.set_ylim(top=None if len(df[df[metric] > 1]) > 0 else 1)
    plt.legend(bbox_to_anchor=(0, 1.07, 1, 0.2), loc='lower left', mode='expand', ncol=3)
    ax.set_xticklabels([probe_map[label.get_text()] for label in ax.get_xticklabels()])
    plt.xticks(rotation=45)
    with out.open('wb') as f:
        plt.savefig(f, bbox_inches='tight')
    plt.close(ax.get_figure())

def overlay_plot(
        df: pd.DataFrame,
        metric: str,
        out: Path,
        struct_only: bool = False
) -> None:
    """
    Plot barplots for individual metrics,
    overlaying main, random and atomic models

    :param df: dataframe with results
    :param metric: metric to plot
    :param out: output path
    :param struct_only: For probes with structural tokens, only plot those
     which were probed on the structural token
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    title = {
        'online_codelength': 'Minimum Description Length',
        'compression': 'Minimum Description Length',
        'accuracy': 'Categorical',
        'r2': 'Regression'}
    # Remove unused metrics
    df = df[[metric, 'model', 'probe_name']]
    df = df.dropna(subset=[metric])
    # Split df into regular, random, and atomic
    regular_subset = df[~df['model'].str.contains('Atom') & ~df['model'].str.contains('Rand')]
    atomic_subset = df[df['model'].str.contains('Atom')]
    random_subset = df[df['model'].str.contains('Rand')]
    # Rename columns
    atomic_subset = atomic_subset.rename(
        columns={
            metric: f'{metric}_atomic',
        }
    )
    atomic_subset['model'] = atomic_subset['model'].str.replace(' Atom', '')
    random_subset = random_subset.rename(
        columns={
            metric: f'{metric}_random'
        }
    )
    random_subset['model'] = random_subset['model'].str.replace(' Rand', '')
    # Merge dataframes
    combined = pd.merge(
        regular_subset,
        atomic_subset,
        on=['probe_name', 'model'],
        how='left'
    )
    combined = pd.merge(
        combined,
        random_subset,
        on=['probe_name', 'model'],
        how='left'
    )

    if struct_only:
        # Remove models that have a struct variant
        has_struct_variant = []
        for model_name in combined['model'].tolist():
            if model_name + '-struct' in combined['model'].tolist():
                has_struct_variant.append(True)
            else:
                has_struct_variant.append(False)
        combined = combined[~pd.Series(has_struct_variant)]

        # Remove '-struct' in model names
        combined['model'] = combined['model'].str.replace('-struct', '')

    # Make new columns for the difference between regular and atomic/random
    combined[f'{metric}_atomic_diff'] = combined[metric] - combined[f'{metric}_atomic']
    combined[f'{metric}_random_diff'] = combined[metric] - combined[f'{metric}_random']

    # Get unique models
    unique_models_without_struct = combined['model'].str.replace('-struct', '').unique()
    # Get colors for each model
    colors = sns.color_palette('colorblind', len(unique_models_without_struct))
    # Create a dictionary mapping model names to colors
    model_colors = dict(zip(unique_models_without_struct, colors))

    # Plot regular probes as bars and random/atomic difference as dots
    # in the same plot
    # sns.set_context('talk')

    fig, ax = plt.subplots()

    bar = sns.barplot(
        x='probe_name',
        y=metric,
        hue='model',
        data=combined,
        order=sorted(combined['probe_name'].unique()),
        errorbar=None,
        ax=ax,
        palette=[
            model_colors[model.replace('-struct', '')]
            for model in combined['model'].unique()
        ]
    )
    try:
        # set hatches
        # This fails when the data for some models is incomplete
        hatches = [
            '///' if model.endswith('-struct') else None
            for model in combined['model']
        ]
        for i, (patch) in enumerate(bar.patches):
            if hatches[i] == '///':
                patch.set_hatch('///')
    except IndexError:
        print('Could not set hatches for barplot, probably because some models are missing data.')

    try:
        sns.stripplot(
            x='probe_name',
            y=f'{metric}_atomic',
            hue='model',
            dodge=True,
            data=combined,
            order=sorted(combined['probe_name'].unique()),
            ax=ax,
            palette='dark:black',
        )
    except KeyError:
        print('Could not plot atomic data, probably because there is no data for atomic models.')

    try:
        sns.stripplot(
            x='probe_name',
            y=f'{metric}_random',
            hue='model',
            dodge=True,
            data=combined,
            order=sorted(combined['probe_name'].unique()),
            ax=ax,
            palette='dark:black',
            marker='D'
        )
    except KeyError:
        print('Could not plot random data, probably because there is no data for random models.')

    handles, labels = ax.get_legend_handles_labels()
    if metric == 'compression':
        ax.set_yscale('log')
    split = out.stem.split('_')[1:]
    ax.set(
        xlabel=None,
        ylabel=' '.join(metric.split('_')).title(),
        title=f"{title[metric]} — {'F1000Research' if split[0].startswith('f1000') else 'Wikipedia'}")
    bottom_ylim = math.floor(df[metric].min()) if len(df[df[metric] < 0]) > 0 else 0
    ax.set_ylim(bottom=bottom_ylim)
    ax.set_ylim(top=None if len(df[df[metric] > 1]) > 0 else 1)
    plt.legend(
        handles=handles[-len(set(list(combined['model']))):],
        labels=labels[-len(combined):],
        bbox_to_anchor=(0, 1.07, 1, 0.2),
        loc='lower left',
        mode='expand',
        ncol=3
    )
    ax.set_xticklabels([probe_map[label.get_text()] for label in ax.get_xticklabels()])
    plt.xticks(rotation=45)
    with out.open('wb') as f:
        plt.savefig(f, bbox_inches='tight')
    plt.close(ax.get_figure())


def table(
        df: pd.DataFrame,
        out: Path,
        sorting: List[str] = None
) -> None:

    def pivot(data: pd.DataFrame):
        data = data.pivot_table(
            index='model',
            columns='probe_name',
            values=['accuracy', 'r2', 'online_codelength', 'compression']
        )
        data.columns = ['/'.join(col).strip() for col in data.columns.values]
        data = data.reset_index()
        return data

    def add_column_suffix(data: pd.DataFrame, suffix: str):
        for col in data.columns:
            if col != 'model':
                data = data.rename(columns={col: f'{col}/{suffix}'})
        return data

    title = {
        'online_codelength': 'Minimum Description Length',
        'compression': 'Minimum Description Length',
        'accuracy': 'Categorical',
        'r2': 'Regression'
    }
    # Split df into regular, random, and atomic
    regular_subset = df[~df['model'].str.contains('Atom') & ~df['model'].str.contains('Rand')]
    regular_subset = pivot(regular_subset)
    regular_subset = add_column_suffix(regular_subset, 'regular')
    atomic_subset = df[df['model'].str.contains('Atom')]
    atomic_subset = pivot(atomic_subset)
    atomic_subset = add_column_suffix(atomic_subset, 'atomic')
    atomic_subset['model'] = atomic_subset['model'].str.replace(' Atom', '')
    random_subset = df[df['model'].str.contains('Rand')]
    random_subset = pivot(random_subset)
    random_subset = add_column_suffix(random_subset, 'random')
    random_subset['model'] = random_subset['model'].str.replace(' Rand', '')
    # Merge dataframes
    combined = pd.merge(
        regular_subset,
        atomic_subset,
        on='model',
        how='left'
    )
    combined = pd.merge(
        combined,
        random_subset,
        on='model',
        how='left'
    )
    if sorting is not None:
        sort_values = {
            name: i
            for i, name in enumerate(sorting)
        }
        combined = combined.sort_values(
            by='model',
            key=lambda x: x.map(sort_values)
        )
    # Write to csv
    combined.to_csv(out / 'table.csv', index=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--location',
        type=str,
        choices=['local', 'cluster']
    )
    parser.add_argument(
        '--model_config',
        type=str,
        action='store'
    )
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--struct_only', type=bool)
    parser.set_defaults(
        location='local',
        model_config='evidence_inference',
        dataset='f1000rd-full',
        struct_only=False
    )
    args = parser.parse_args()

    model_configs = {
        'led': [
            'led-base-16384/003led'
        ]
    }

    local_base = Path(__file__).parent.parent / 'data' / 'probing_results' / 'in_n_out'

    if args.location == 'local':
        base = local_base / args.dataset
    else:
        raise NotImplementedError

    results = load_results(
        base,
        model_configs[args.model_config],
        args.location
    )
    width, height = rcParams['figure.figsize']

    df = probe_results(results)
    sorting = [
        model_map[model] for model in model_configs[args.model_config]
    ]
    table(df, local_base, sorting=sorting)
    if len(model_map) > 1:
        rcParams['figure.figsize'] = width * 2, height
    else:
        rcParams['figure.figsize'] = width, height
    for metric in ['online_codelength', 'compression', 'accuracy', 'r2']:
        plot_probes(
            df,
            metric,
            local_base / 'plots' / f'probes_{args.dataset}_{metric}.png'
        )
        if args.struct_only:
            out_path = local_base / 'plots' / f'probes_{args.dataset}_{metric}_struct_only_combined.png'
        else:
            out_path = local_base / 'plots' / f'probes_{args.dataset}_{metric}_combined.png'
        overlay_plot(
            df,
            metric,
            out_path,
            struct_only=args.struct_only
        )