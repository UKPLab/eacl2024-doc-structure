import json
from pathlib import Path

from intertext_graph.itgraph import IntertextDocument

def main(data_dir_path: Path):
    for split_name in ['train', 'dev', 'test']:
        deep_data = []
        with open(data_dir_path / f'deep-{split_name}.jsonl') as file:
            for line in file:
                deep_data.append(IntertextDocument._from_json(json.loads(line)))

        shallow_data = []
        with open(data_dir_path / f'shallow-{split_name}.jsonl') as file:
            for line in file:
                shallow_data.append(IntertextDocument._from_json(json.loads(line)))

        for deep_itg, shallow_itg in zip(deep_data, shallow_data):
            for deep_node, shallow_node in zip(deep_itg.unroll_graph(), shallow_itg.unroll_graph()):
                if not deep_node.ix == shallow_node.ix:
                    print(f'Ix mismatch for {deep_node.ix}, {shallow_node.ix}')
                    continue
                if not deep_node.ntype == shallow_node.ntype:
                    print(f'Type Mismatch for {deep_node.ix}')
                    continue

                if deep_node.ntype in ['p', 'article-title']:
                    if not deep_node.content == shallow_node.content:
                        print(f'Content mismatch for {deep_node.ix}')
                elif deep_node.ntype in ['title']:
                    if not deep_node.content == shallow_node.content.split(' ::: ')[-1]:
                        print(f'section title mismatch for {deep_node.ix}')

if __name__ == '__main__':
    DATA_DIR_PATH = Path('/path/to/qasper-itg-data')

    main(DATA_DIR_PATH)