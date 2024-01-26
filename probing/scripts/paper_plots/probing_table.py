from pathlib import Path
from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from scripts.probing_results import load_results, probe_results


non_infused_models = {
    'LED': {
        'led-base-16384/003led': 'LED',
    },
    'LongT5': {
        'long-t5-tglobal-base/031_t5_lr': 'LongT5',
    }
}

infused_models = {
    'LED': {
        'led-base-16384/045_F1000_pretrained_LED_is-node-boundaries_probe_struct-f6da-ae8c': 'tok-boundaries',
        'led-base-16384/046_F1000_pretrained_LED_is-node-depths_probe_struct-f56f-5dfe': 'tok-depth',
        'led-base-16384/047_F1000_pretrained_LED_is-node-types_probe_struct-040a-daf9': 'tok-type',
        'led-base-16384/041_F1000_pretrained_LED_pe-node-depths-adb0-6d4a': 'emb-depth',
        'led-base-16384/042_F1000_pretrained_LED_pe-node-types-a9d5-7af4': 'emb-type',
        'led-base-16384/048_F1000_pretrained_LED_pe-node-depths-is-node-types_probe_struct-a73b-bd0b': 'emb-depth-tok-type',
        'led-base-16384/049_F1000_pretrained_LED_pe-node-types-is-node-types_probe_struct-f298-6e7c': 'emb-type-tok-type',
        'led-base-16384/049_F1000_pretrained_LED_pe-node-types-is-node-depths_probe_struct-9000-7949': 'emb-type-tok-depth',
        'led-base-16384/050_F1000_pretrained_LED_pe-node-depths-is-node-depths_probe_struct-d38b-c0a4': 'emb-depth-tok-depth',
    },
    'LongT5': {
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

probe_short_names = {
    'node_type': 'Nod',
    'sibling': 'Sib',
    'ancestor': 'Anc',
    'position': 'Pos',
    'parent_predecessor': 'Par',
    'tree_depth': 'Tre',
    'structural': 'Str'
}


metric = 'accuracy'

data_path = Path('../../data/probing_results/in_n_out/f1000rd-full')
out_dir_path = Path('../../out/plots')
out_name = 'probing_plot.pdf'

def main():

    data = load_data()
    write_table(data)


def load_data():
    data = {}
    for model in non_infused_models:
        raw_data = load_results(
            data_path,
            list(non_infused_models[model].keys()),
            'local'
        )
        df = probe_results(raw_data)

        df = pd.pivot_table(
            df,
            values=metric,
            index=['model'],
            columns=['probe_name']
        )
        df = df.reset_index()
        data[model] = df

    for model in infused_models:
        raw_data = load_results(
            data_path,
            list(infused_models[model].keys()),
            'local'
        )
        df = (probe_results(raw_data))

        # Remove random and atomic data
        df = df[~df['model'].str.contains('Rand')]
        df = df[~df['model'].str.contains('Atomic')]

        df = pd.pivot_table(
            df,
            values=metric,
            index=['model'],
            columns=['probe_name']
        )

        df = df.reset_index()

        # sort values
        order = infused_models[model].values()
        sorting_indices = list(range(len(order)))
        sorting_series = pd.Series(sorting_indices, index=order)
        df['sorting'] = df['model'].map(sorting_series)
        df = df.sort_values('sorting')
        df = df.drop(columns=['sorting'])
        data[model] = pd.concat([data[model], df], ignore_index=True)



    return data


def write_table(data: Dict[str, pd.DataFrame]):
    df = list(data.values())[0]
    # Append all other dataframes
    for df_to_append in list(data.values())[1:]:
        df = pd.concat([df, df_to_append], ignore_index=True)

    new_column_names = {
        'model': ''
    }
    new_column_names.update(
        probe_short_names
    )
    df = df.rename(columns=new_column_names)
    # sort columns
    order = [''] + list(probe_short_names.values())
    df = df.reindex(order, axis=1)
    pass

    df = df.set_index('')
    df[list(probe_short_names.values())] = df[list(probe_short_names.values())] * 100

    latex = df.to_latex(
        float_format="%.2f",
        index=True,
        escape=False
    )

    print(latex)



if __name__ == '__main__':
    main()

