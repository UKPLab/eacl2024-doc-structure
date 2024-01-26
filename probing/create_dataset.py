from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree

from probing_kit.dataset.edge_probing import EdgeProbingPipeline
from probing_kit.dataset.infusion import InfusionPipeline
from probing_kit.dataset.probing_tasks import ProbingTaskPipeline


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', action='store', type=str)
    parser.add_argument('--max_length', action='store', type=int)
    parser.add_argument('--infusion_method', action='store', type=str)
    parser.add_argument('--with_closing_tags', action='store_true')
    parser.add_argument('--probe_structure_tags', action='store_true')
    parser.add_argument('--no-recover', dest='recover', action='store_false')
    parser.set_defaults(
        model='google/long-t5-tglobal-base',
        max_length=16384,
        infusion_method=None,
        with_closing_tags=False,
        probe_structure_tags=False,
        recover=True)
    args = parser.parse_args()
    dataset_dir = Path(__file__).parent / 'out' / 'f1000rd-full' / str(args.model).rsplit('/', 1)[-1]
    for path in dataset_dir.iterdir():
        if path.stem == 'preprocessed':
            continue
        elif path.is_dir() and not args.recover:
            rmtree(path, ignore_errors=True)
    ProbingTaskPipeline.run_default_config(dataset_dir, args.infusion_method)
    if args.infusion_method is not None:
        InfusionPipeline.run_default_config(dataset_dir, args.model, args.max_length, args.infusion_method, args.with_closing_tags, args.probe_structure_tags, args.recover)
    EdgeProbingPipeline.run_default_config(dataset_dir, args.model, args.infusion_method, args.with_closing_tags, args.probe_structure_tags, args.recover)
