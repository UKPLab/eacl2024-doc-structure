from argparse import ArgumentParser
from pathlib import Path

from probing_kit.experiment import run

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('version', action='store', type=str)
    parser.add_argument('--in_n_out_dir_path', action='store', type=Path)
    parser.add_argument('--dataset', action='store', type=str)
    parser.add_argument('--model', action='store', type=str)
    parser.add_argument('--max_length', action='store', type=int)
    parser.add_argument('--probes', action='store', nargs='+', type=str)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--atomic', action='store_true')
    parser.add_argument('--no-recover', dest='recover', action='store_false')
    parser.add_argument('--train_parameters', action='store_true')
    parser.add_argument('--state_dict_path', action='store', type=Path)
    parser.add_argument('--infusion_method', action='store', type=str)
    parser.add_argument('--with_closing_tags', action='store_true')
    parser.add_argument('--probe_structure_tags', action='store_true')
    parser.add_argument('--position_embeddings_mode', action='store', type=str)
    parser.add_argument('--global_attention_mode', action='store', type=str)
    parser.add_argument('--lr', action='store', type=float)
    parser.add_argument('--batch_size', action='store', type=int)
    parser.set_defaults(
        version='001',
        in_n_out_dir_path=None,
        dataset='f1000rd-full',
        model='google/long-t5-tglobal-base',
        max_length=16384,
        probes=[
            'node_type',
            'structural',
            'tree_depth',
            'sibling',
            'ancestor',
            'position',
            'parent_predecessor'
        ],
        random=False,
        atomic=False,
        infusion_method=None,
        with_closing_tags=False,
        position_embeddings_mode='vanilla',
        global_attention_mode='vanilla',
        probe_structure_tags=False,
        recover=True,
        train_parameters=False,
        state_dict_path=None,
        lr=0.1,
        batch_size=4)
    args = parser.parse_args()
    run(
        version=args.version,
        in_n_out_dir_path=args.in_n_out_dir_path,
        dataset=args.dataset,
        pretrained_model_name_or_path=args.model,
        max_length=args.max_length,
        probes=args.probes,
        random_weights=args.random,
        atomic=args.atomic,
        method=args.infusion_method,
        with_closing=args.with_closing_tags,
        position_embeddings_mode=args.position_embeddings_mode,
        global_attention_mode=args.global_attention_mode,
        probe_struct=args.probe_structure_tags,
        recover=args.recover,
        train_parameters=args.train_parameters,
        state_dict_path=args.state_dict_path,
        lr=args.lr,
        batch_size=args.batch_size)
