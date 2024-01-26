import json
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.nn.functional import softmax

MODELS = {
    'led-base-16384/003led': 'LED',
    'long-t5-tglobal-base/031_t5_lr': 'LongT5'
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

def main():
    data_path = Path('../../data/probing_results/in_n_out/f1000rd-full')
    out_path = Path('../../out/plots')

    data = load_results(
        data_path,
    )
    plot_layer_utilization(data, out_path)

def load_results(
        data_path,
):
    results = {}
    # Load results jsons
    for model in MODELS:
        for variant in '', '-atomic', '-random':
            path = data_path / (model + variant)
            if os.path.isdir(path):
                name = '/'.join(path.parts[-2:])
                results[name] = []
                for probe in PROBE_PRETTY_NAMES.keys():
                    probe_path = path / probe
                    if os.path.exists(probe_path / 'probing.json'):
                        with open(probe_path / 'probing.json') as f:
                            results[name].append(json.load(f))
                    else:
                        if model.startswith('led-base-16384'):
                            num_layers = 6
                        elif model.startswith('long-t5-tglobal-base'):
                            num_layers = 12
                        results[name].append({
                            'probe': {
                                'probe_name': probe
                            },
                            'scalar_mix': {
                                str(i): 1 / num_layers
                                for i in range(num_layers)
                            }
                        })

    # Extract layer utilization data
    layer_utilization = {}
    for model, data in results.items():
        # Make model name pretty
        if model.endswith('-random'):
            key = model.replace('-random', '')
            model_pretty_name = MODELS[key] + ' Rand'
        elif model.endswith('-atomic'):
            key = model.replace('-atomic', '')
            model_pretty_name = MODELS[key] + ' Atom'
        else:
            model_pretty_name = MODELS[model]



        # Make list of scalars for each probe
        layer_utilization[model_pretty_name] = {
            probe_pretty_name: []
            for probe_pretty_name in PROBE_PRETTY_NAMES.values()
        }
        layer_utilization[model_pretty_name]['Layer'] = []

        # Extract layer utilization data
        for probe_data in data:
            probe_name = probe_data['probe']['probe_name']
            probe_pretty_name = PROBE_PRETTY_NAMES[probe_name]
            if len(layer_utilization[model_pretty_name]['Layer']) == 0:
                # If layer numbers were not extracted yet, extract them
                layer_utilization[model_pretty_name]['Layer'] = [
                    int(i) for i in probe_data['scalar_mix'].keys()
                ]
            for value in probe_data['scalar_mix'].values():
                layer_utilization[model_pretty_name][probe_pretty_name].append(value)

        layer_utilization[model_pretty_name] = pd.DataFrame(layer_utilization[model_pretty_name])

    return layer_utilization

def plot_layer_utilization(
        data,
        out_path
):
    cm = 1 / 2.54
    n_models = len(data)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(16 * cm, 5 * cm)
    )

    titles = []
    for i, model_name in enumerate(data):
        titles.append(model_name)
        plot_id = i
        heatmap_data = data[model_name]

        # Add 1 to layer index
        heatmap_data['Layer'] = heatmap_data['Layer'] + 1
        # Make layer index the index
        heatmap_data = heatmap_data.set_index('Layer')

        # Normalize columns by softmax
        heatmap_data = heatmap_data.apply(normalize_series_softmax, axis=0)

        add_colorbar = True
        # Add colorbar only to upper right plot
        if i + 1 != n_models:
            add_colorbar = False



        # transpose data
        heatmap_data = heatmap_data.transpose()

        heatmap = sns.heatmap(
            data=heatmap_data,
            fmt='s',
            center=0.3,
            vmax=1,
            vmin=0,
            cmap='gray_r',
            ax=axes[plot_id],
            cbar=add_colorbar,
        )


        ax = axes[plot_id]
        if ax.collections[0].colorbar is not None:
            ax.collections[0].colorbar.set_ticks([0, 0.5, 1])
            ax.collections[0].colorbar.set_ticklabels(['0', '0.5', '1'])

        if 'Rand' in model_name:
            title = 'Rand'
        elif 'Atom' in model_name:
            title = 'Atom'
        else:
            title = model_name
        ax.set_title(title)
        ax.tick_params(length=0)
        if i > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        # Show x axis title only for middle plot
        ax.set_xlabel('')

        if 'LED' in model_name:
            tick_interval = 2
        elif 'LongT5' in model_name:
            tick_interval = 3

        ax.set_xticks(
            [
                i + 0.5 for i in range(len(heatmap_data.columns))
                if i % tick_interval == 0
            ],

        )
        ax.set_xticklabels(
            [
                i + 1 for i in range(len(heatmap_data.columns))
                if i % tick_interval == 0
            ]
        )

    plt.tight_layout()
    plt.savefig(out_path / 'layer_util.pdf')
    plt.clf()

def normalize_series_softmax(series):
    values = torch.from_numpy(series.to_numpy())
    softmaxed = softmax(values, dim=0)
    return softmaxed.tolist()

if __name__ == '__main__':
    main()