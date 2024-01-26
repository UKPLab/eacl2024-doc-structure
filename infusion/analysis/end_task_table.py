import json
import os
import re
from pathlib import Path
from typing import Dict, Union

import pandas as pd

TASK_SPECIFIC_RESULT_KEYS = {
    'QASPER': [
        'answer_f1',
        'evidence_f1'
    ],
    'Evidence Inference': [
        'classification_macro_f1_score',
        'evidence_detection_f1_score',
    ]
}

METRIC_PRETTY_NAMES = {
    'answer_f1': 'Ans',
    'evidence_f1': 'Evi',
    'classification_macro_f1_score': 'Cla',
    'evidence_detection_f1_score': 'Evi'
}

TASK_PRETTY_NAMES = {
    'QASPER': 'QAS',
    'Evidence Inference': 'EvI'
}

def main():
    results_path = Path('../data/results')
    data = load_data(results_path=results_path)
    data=data*100
    # Get latex table with results, rounded to 2 decimal places
    # and with multirow and multicolumns
    # Do not show the model and config columns
    data.index.name = None
    data.columns.name = None
    data.index.names = [None]
    latex = data.to_latex(
        float_format="%.2f",
        multirow=True,
        multicolumn=True,
        multicolumn_format='c|',
        index=True,
        escape=False
    )


    # latex = data.to_latex(
    #     float_format="%.2f",
    #     multirow=True,
    #     multicolumn=True,
    #     multicolumn_format='c',
    # )
    print(latex)

def load_data(results_path: Path) -> pd.DataFrame:

    task_hash_mapping_paths = {
        'LED': {
            'QASPER': '../data/hash_description_mapping_qasper.csv',
            'Evidence Inference': '../data/hash_description_mapping_evidence_inference.csv'
        },
        'LongT5': {
            'QASPER': '../data/hash_description_mapping_longt5_qasper.csv',
            'Evidence Inference': '../data/hash_description_mapping_longt5_evidence_inference.csv'
        }
    }

    dfs = []
    for model_name in task_hash_mapping_paths:
        hash_mappings = task_hash_mapping_paths[model_name]
        dfs.append(load_data_for_model(
            model_name=model_name,
            results_path=results_path,
            task_hash_mapping_paths=hash_mappings
        ))

    # Merge all dataframes
    df = dfs[0]
    for df_right in dfs[1:]:
        df = df.merge(df_right, on=[('', '', 'config')], how='left')

    return df





def load_data_for_model(
        model_name: str,
        results_path: Path,
        task_hash_mapping_paths: Dict[str, str]
) -> pd.DataFrame:
    rotate = lambda x: '\\rotatebox{90}{' + x + '}'

    results_filenames = [
        str(filename)
        for filename in os.listdir(results_path)
    ]
    dfs = []
    for task_name in task_hash_mapping_paths.keys():
        results = {
            ('', '', 'config'): []
        }
        results.update({
            (model_name, TASK_PRETTY_NAMES[task_name], METRIC_PRETTY_NAMES[metric_name]): []
            for metric_name in TASK_SPECIFIC_RESULT_KEYS[task_name]
        })

        hash_mapping = pd.read_csv(task_hash_mapping_paths[task_name])

        for hash, config in zip(hash_mapping['Test Hash'], hash_mapping['Description']):

            regex = f'{task_name}.*{hash}.json'
            results_json = None
            for filename in results_filenames:
                if re.match(regex, filename):
                    with open(results_path / filename) as f:
                        results_json = json.load(f)
                        break
            if results_json is not None:
                results[('', '', 'config')].append(config)
                for metric_name in TASK_SPECIFIC_RESULT_KEYS[task_name]:
                    results[(model_name, TASK_PRETTY_NAMES[task_name], METRIC_PRETTY_NAMES[metric_name])].append(results_json['test_result'][metric_name])

        dfs.append(pd.DataFrame(
            results,
            columns=pd.MultiIndex.from_tuples(results.keys())
        ))
    df = dfs[0]
    for df_right in dfs[1:]:
        df = df.merge(df_right, on=[('', '', 'config')], how='left')

    df = df.groupby([('', '', 'config')], sort=False).mean()
    return df


if __name__ == '__main__':
    main()