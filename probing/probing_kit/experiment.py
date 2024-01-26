from pathlib import Path
from shutil import rmtree
from typing import List, Union

from probing_kit.dataset.edge_probing import EdgeProbingPipeline
from probing_kit.dataset.probing_tasks import ProbingTaskPipeline
from probing_kit.training import run_training_loop

# Check if relpos_graph repo is available (needed for position embeddings)
try:
    from infusion.structformer.input_sequence import *
    infusion_available = True
except (ImportError, ModuleNotFoundError):
    infusion_available = False

def run(
        version: Union[int, str],
        in_n_out_dir_path: Path = None,
        dataset: str = 'f1000rd-full',
        pretrained_model_name_or_path: Union[str, Path] = 'allenai/longformer-base-4096',
        max_length: int = 4096,
        probes: List[str] = None,
        random_weights: bool = False,
        atomic: bool = False,
        method: str = None,
        with_closing: bool = False,
        position_embeddings_mode: str = 'vanilla',
        global_attention_mode: str = 'vanilla',
        probe_struct: bool = False,
        recover: bool = True,
        train_parameters: bool = False,
        state_dict_path: Path = None,
        lr: float = 0.1,
        batch_size: int = 4) -> None:
    assert not random_weights or not atomic
    # Dataset creation

    if position_embeddings_mode != 'vanilla':
        assert infusion_available, "Position embeddings mode is not vanilla, but infusion is not available."
    if global_attention_mode != 'vanilla':
        assert infusion_available, "Global attention mode is not vanilla, but infusion is not available."

    if in_n_out_dir_path:
        dataset_dir = in_n_out_dir_path
    else:
        dataset_dir = Path(__file__).parent.parent / 'out' / dataset / str(pretrained_model_name_or_path).rsplit('/', 1)[-1]
    if not recover:
        for path in dataset_dir.iterdir():
            if path.stem == 'preprocessed':
                continue
            elif path.is_dir():
                rmtree(path, ignore_errors=True)
    infusion_path = f"{f'_{method}' if method is not None else ''}{'_w_closing' if with_closing else ''}{'_probe_struct' if probe_struct else ''}"
    if not (dataset_dir / f"intertext_graph{infusion_path if method is not None else ''}").exists():
        ProbingTaskPipeline.run_default_config(dataset_dir)
        EdgeProbingPipeline.run_default_config(dataset_dir, pretrained_model_name_or_path)
    # Run probes
    if probes is None:
        probes = [
            'node_type',
            'structural',
            'tree_depth',
            'sibling',
            'ancestor',
            'position',
            'parent_predecessor']
    run_training_loop(
        in_n_out=dataset_dir / f"{version}{('-random' if random_weights else '-atomic' if atomic else '')}",
        dataset=dataset_dir,
        probes=probes,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        max_length=max_length,
        random_weights=random_weights,
        atomic=atomic,
        method=method,
        position_embeddings_mode=position_embeddings_mode,
        global_attention_mode=global_attention_mode,
        with_closing=with_closing,
        probe_struct=probe_struct,
        recover=recover,
        train_parameters=train_parameters,
        state_dict_path=state_dict_path,
        lr=lr,
        batch_size=batch_size)
