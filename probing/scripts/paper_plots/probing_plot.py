from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from scripts.probing_results import load_results, probe_results


models = {
    'led-base-16384/003led': 'LED',
    'long-t5-tglobal-base/031_t5_lr': 'LongT5',
}


def main():
    data_path = Path('../../data/probing_results/in_n_out/f1000rd-full')
    out_dir_path = Path('../../out/plots')
    out_name = 'probing_plot.pdf'

    data = load_results(
        data_path,
        list(models.keys()),
        'local'
    )
    df = probe_results(data)

    infused_probing_plot(df, out_dir_path / out_name)




def infused_probing_plot(
        data: pd.DataFrame,
        out: Path,
):
    classification_metric = 'accuracy'
    regression_metric = 'r2'
    baseline = 'vanilla'
    classification_probes = {
        'node_type': 'Node\nType',
        'sibling': 'Sibling',
        'ancestor': 'Ancestor',
        'position': 'Position',
        'parent_predecessor': 'Parent\n Predecessor',
        'tree_depth': 'Tree\nDepth',
        'structural': 'Structural'
    }
    regression_probes = {

    }

    # Get score as classification score or regression score
    # depending on probe
    data['score'] = 0
    # Get classification scores
    data.loc[
        data['probe_name'].isin(classification_probes),
        'score'
    ] = data.loc[
        data['probe_name'].isin(classification_probes),
        classification_metric
    ]
    # Get regression scores
    data.loc[
        data['probe_name'].isin(regression_probes),
        'score'
    ] = data.loc[
        data['probe_name'].isin(regression_probes),
        regression_metric
    ]

    sns.set_context('paper')
    # Set up plot
    plt.rcParams['hatch.linewidth'] = 0
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        figsize=(16*cm, 3.85*cm),
    )

    plot = sns.barplot(
        data=data,
        x='probe_name',
        y='score',
        hue='model',
        palette=[
            'lightgray', 'lightgray', 'lightgray',
            'gray', 'gray', 'gray'
        ],
        errorbar=None,
        ax=ax,
        order=(
            list(classification_probes.keys())
            + list(regression_probes.keys())
        )
    )
    # Set hatches
    hatches = []
    for model in data['model']:
        if model.endswith('Rand'):
            hatches.append('///')
        elif model.endswith('Atom'):
            hatches.append('...')
        else:
            hatches.append(None)
    for i, bar in enumerate(plot.patches):
        bar.set_edgecolor('black')
        if i < len(hatches):
            if hatches[i] is not None:
                bar.set_hatch(hatches[i])


    if len(regression_probes) > 0:
        # Vertical line between classification and regression probes
        ax.axvline(
            x=len(classification_probes) - 0.5 ,
            color='black',
            linewidth=1
        )
        # secondary axis for r2 label
        ax2 = ax.twinx()
        ax2.set_ylim(bottom=0.4)
        ax2.set_ylabel('R2', labelpad=10)

    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles=handles[-len(set(list(data['model']))):],
        labels=labels[-len(data):],
        bbox_to_anchor=(1, -0.5, 0.3, 0.1),
        loc='lower left',
        # mode='expand',
        ncol=1,
        fontsize=8
    )
    ax.set_ylabel('Accuracy', fontdict={
            'fontsize': 8
        })
    ax.set_yticklabels(
        labels=[0, 1],
        fontdict={
            'fontsize': 8
        }
    )
    ax.set_xticklabels(
        list(classification_probes.values())
        + list(regression_probes.values()),
        fontdict={
            'fontsize': 8
        }
    )
    plt.xlabel('')
    # plt.xticks(rotation=45)
    # plt.rc('font', size=0)

    # Scale plot because legend will stick out otherwise
    # left, bottom, width, height
    plt.tight_layout(
    #    rect=[0, 0, 0.8, 1.10]
    )
    plt.savefig(out, bbox_inches='tight')


if __name__ == '__main__':
    main()