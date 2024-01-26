from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from scripts.probing_results import load_results, probe_results


models = {
    'LED': {
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
    },
    'LongT5': {
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
}

METRIC_PRETTY_NAMES = {
    'accuracy': 'Accuracy',
    'r2': 'R²',
    'compression': 'Compression'
}


def main():
    data_path = Path('../../data/probing_results/in_n_out/f1000rd-full')
    out_dir_path = Path('../../out/plots')
    out_name = 'infused_probing_plot.pdf'

    results = {}
    for model in models:
        results[model] = load_results(
            data_path,
            list(models[model].keys()),
            'local'
        )
        results[model] = (probe_results(results[model]))

        # Remove random and atomic data
        results[model] = results[model][~results[model]['model'].str.contains('Rand')]
        results[model] = results[model][~results[model]['model'].str.contains('Atom')]

    # Set up plot
    # sns.set_theme(style='whitegrid')
    plt.rcParams.update({'font.size': 8})
    cm = 1 / 2.54
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(16 * cm, 5 * cm),
    )

    legend = None
    for i, model in enumerate(models):
        is_top = False
        is_bottom = False
        if i == 0:
            is_top = True
        elif i == len(models) - 1:
            is_bottom = True
        lgd = infused_probing_plot(
            results[model],
            axes[i],
            model,
            is_top=is_top,
            is_bottom=is_bottom
        )
        if lgd is not None:
            legend = lgd


    fig.tight_layout(rect=[0, 0, 0.74, 1.10])
    plt.savefig(out_dir_path / out_name, bbox_extra_artists=(legend, ), bbox_inches='tight')




def infused_probing_plot(
        data: pd.DataFrame,
        ax: plt.Axes,
        model: str,
        is_top: bool = False,
        is_bottom: bool = False
):
    struct_only = True
    classification_metric = 'accuracy'
    regression_metric = 'r2'
    baseline = 'vanilla'
    plot_ratios = True
    classification_probes = {
        'node_type': 'Nod',
        'sibling': 'Sib',
        'ancestor': 'Anc',
        'position': 'Pos',
        'parent_predecessor': 'Par',
        'tree_depth': 'Tre',
        'structural': 'Str'
    }
    regression_probes = {

    }

    # Get score as classification score or regression score
    # depending on probe
    data['score'] = 0

    data['baseline'] = 0
    # Get classification scores
    data.loc[
        data['probe_name'].isin(classification_probes),
        'score'
    ] = data.loc[
        data['probe_name'].isin(classification_probes),
        classification_metric
    ]
    for probe in classification_probes:
        try:
            data.loc[
                data['probe_name'] == probe,
                'baseline'
            ] = data.loc[(data['probe_name'] == probe) & (data['model'] == baseline), classification_metric].values[0]
        except IndexError:
            pass
    if len(regression_probes) > 0:
        # Get regression scores
        data.loc[
            data['probe_name'].isin(regression_probes),
            'score'
        ] = data.loc[
            data['probe_name'].isin(regression_probes),
            regression_metric
        ]
        for probe in regression_probes:
            data.loc[
                data['probe_name'] == probe,
                'baseline'
            ] = data.loc[
                (data['model'] == baseline)
                & (data['probe_name'] == probe),
                classification_metric
            ].values[0]

    sns.set_context('paper')

    data['score'] = data['score'] - data['baseline']

    # Remove '-struct' suffix
    data['model'] = data['model'].apply(lambda x: x.replace('-struct', ''))

    # Plot everything but vanilla
    plot = sns.barplot(
        data=data[data['model'] != baseline],
        x='probe_name',
        y='score',
        hue='model',
        palette='colorblind',
        errorbar=None,
        ax=ax,
        order=(
            list(classification_probes.keys())
            + list(regression_probes.keys())
        )
    )
    # Show borders
    plt.setp(ax.patches, linewidth=1, edgecolor='black')
    if len(regression_probes) > 0:
        # Vertical line between classification and regression probes
        ax.axvline(
            x=len(classification_probes) - 0.5 ,
            color='black',
            linewidth=1
        )
        ax2 = ax.twinx()
        ax2.set_ylim(bottom=0.4)
        ax2.set_ylabel(METRIC_PRETTY_NAMES[regression_metric], labelpad=10)
        ax2.set_yticks([])
    # Logarithmic scale for y axis
    ax.set_ylabel(f'{model}\n $\Delta$ {METRIC_PRETTY_NAMES[classification_metric]}')
    ax.set_xticklabels(
        list(classification_probes.values())
        + list(regression_probes.values())
    )
    # ax.set_ylim(bottom=0.3)
    plt.xlabel('')
    # plt.xticks(rotation=45)
    # secondary axis for r2 label


    # Remove legend
    ax.legend_.remove()
    ax.set_xlabel(None)

    # Set yticks to 0.0 and 0.1
    ax.set_yticks([0.0, 0.1])

    if not is_bottom:
        # Remove x ticks and labels
        ax.set_xticklabels([])
        ax.set_xticks([])

    lgd = None
    if is_top:
        handles, labels = ax.get_legend_handles_labels()
        lgd = plt.figlegend(
            handles=handles[-len(set(list(data['model']))):],
            labels=labels[-len(data):],
            bbox_to_anchor=(0.72, 0.16, 0.28, 0.2),
            loc='lower left',
            mode='expand',
            ncol=1,
            fontsize=8
        )

    return lgd




if __name__ == '__main__':
    main()