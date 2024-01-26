import bz2
from datetime import date
import json
from pathlib import Path
from typing import Dict, List, Union, OrderedDict, Any

from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from probing_kit.embedder import ProbingKitEmbedder
import requests
from tempfile import TemporaryDirectory

import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_dir(path: Union[Path, str], suffix: str) -> List[Path]:
    if isinstance(path, str):
        path = Path(path)
    return [file for file in path.iterdir() if file.suffix == suffix]


def read_wiki(path: Union[Path, str], timestamp: date) -> List[str]:
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        data = {}
        dump = f'https://dumps.wikimedia.org/other/pageview_complete/monthly/{timestamp.year}/{timestamp.year}-{timestamp.month}/pageviews-{timestamp.year}{timestamp.month}-user.bz2'
        with TemporaryDirectory() as tmpdir:
            pageviews = Path(tmpdir) / 'pageviews.txt'
            with requests.get(dump, stream=True) as response:
                response.raise_for_status()
                decomp = bz2.BZ2Decompressor()
                total = int(response.headers.get('Content-Length', 0))
                progress_bar = tqdm(total=total, unit='B', unit_scale=True, desc='loading Wikipedia pageviews')
                with pageviews.open('wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        progress_bar.update(len(chunk))
                        f.write(decomp.decompress(chunk))
            with pageviews.open('r') as f:
                for line in f:
                    wiki_code, *rest, _, tag, daily_total, _ = line.split()
                    if wiki_code == 'en.wikipedia' and tag in ['desktop', 'mobile-web']:
                        article_title = ' '.join(rest)
                        if article_title not in data:
                            data[article_title] = 0
                        data[article_title] += int(daily_total)
            with path.open('w') as f:
                json.dump({key: value for key, value in sorted(data.items(), key=lambda item: item[1], reverse=True)}, f)
    with path.open('r') as f:
        data = json.load(f)
        return list(data.keys())


def train_test_dev_split(files: List[Path]) -> Dict[str, List[Path]]:
    train, test = train_test_split(files, test_size=0.4, random_state=0)
    test, dev = train_test_split(test, test_size=0.5, random_state=0)
    return {'train': train, 'test': test, 'dev': dev}


def probe_spans() -> Dict[str, int]:
    return {
        'node_type': 1,
        'structural': 2,
        'tree_depth': 2,
        'sibling': 2,
        'ancestor': 2,
        'position': 1,
        'parent_predecessor': 2
    }


def structure_tokens(method: str, with_closing: bool) -> List[str]:
    if method == 'node_boundaries':
        tokens = ['<node>']
    elif method == 'node_types':
        tokens = ['<p>', '<title>', '<article-title>', '<abstract>']
    elif method == 'node_depth':
        tokens = [f'<node-{depth + 1}>' for depth in range(20)]
    else:
        raise AssertionError
    if with_closing:
        tokens_with_closing = []
        for token in tokens:
            tokens_with_closing.extend([token, f'</{token[1:]}'])
        return tokens_with_closing
    else:
        return tokens


# ***********************************************************************
# Added by

def load_token_embedder_state_dict(
        token_embedder: ProbingKitEmbedder,
        state_dict_path: Path
) -> PretrainedTransformerEmbedder:
    """Load the trained weights from a saved state_dict into a
    PreTrainedTransformerEmbedder model. This function assumes that
    the state_dict at state_dict_path was automatically saved during
    probing"""

    cuda = torch.cuda.is_available()
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # load new state_dict
    new_state_dict: Union[OrderedDict, Dict] = torch.load(
        state_dict_path,
        map_location=torch.device(device)
    )

    # Check the type of state dict to see how it needs to be loaded
    if 'epoch' in new_state_dict.keys():
        # State dict was saved in structure benchmark
        # Check if it is LED or longformer
        if token_embedder.transformer_model._get_name().startswith('LED'):
            new_state_dict = prepare_led_state_dict_from_structure_benchmark(new_state_dict, token_embedder)
        elif token_embedder.transformer_model._get_name().startswith('Longformer'):
            new_state_dict = prepare_longformer_state_dict_from_structure_benchmark(new_state_dict)
        elif token_embedder.transformer_model._get_name().startswith('LongT5'):
            new_state_dict = prepare_longt5_state_dict_from_structure_benchmark(new_state_dict, token_embedder)

    else:
        # Assume that state dict was saved during probing

        # check if scalar mix parameters are needed
        if token_embedder._scalar_mix is None:
            # Filter keys in new_state_dict for those that are relevant
            # for the token_embedder (excluding the _scalar_mix parameters)
            new_state_dict = {
                '.'.join(k.split('.')[1:]): v
                for k, v in new_state_dict.items()
                if k.startswith('_token_embedder.transformer_model')
            }
        else:
            # Filter keys in new_state_dict for those that are relevant
            # for the token_embedder (including the _scalar_mix parameters)
            new_state_dict = {
                '.'.join(k.split('.')[1:]): v
                for k, v in new_state_dict.items()
                if k.startswith('_token_embedder')
        }

    # check if scalar mix parameters are needed but not there
    if (token_embedder._scalar_mix is not None) and not any(
        k.startswith('_scalar_mix')
        for k in new_state_dict
    ):
        # Initialize scalar mix parameters as 0. (default in allennlp.modules.ScalarMix)
        scalar_mix_parameters = {
            k: torch.tensor([0.])
            for k in token_embedder.state_dict()
            if k.startswith('_scalar_mix.scalar_parameters')
        }
        # add gamma
        scalar_mix_parameters['_scalar_mix.gamma'] = torch.tensor([1.0])

        # add scalar mix parameters to state_dict
        new_state_dict.update(scalar_mix_parameters)

    token_embedder.load_state_dict(new_state_dict)

    return token_embedder

def prepare_led_state_dict_from_structure_benchmark(
        state_dict: Dict[str, Any],
        token_embedder: PretrainedTransformerEmbedder
) -> Dict[str, torch.Tensor]:
    """Prepare a state dict that was saved with the structure benchmark
    with LED to be loaded into a probing model."""
    state_dict = state_dict['state_dict']

    # Parameter naming differs between models
    if any(k.startswith('led_model.led.encoder') for k in state_dict.keys()):
        state_dict = {
            f'transformer_model.{".".join(k.split(".")[2:])}': v
            for k, v in state_dict.items()
            if k.startswith('led_model.led.')
        }
    else:
        state_dict = {
            f'transformer_model.{".".join(k.split(".")[1:])}': v
            for k, v in state_dict.items()
            if k.startswith('led_model.')
        }

    # Hard code the names of resized parameters
    led_resized_parameter_names = [
        'transformer_model.shared.weight',
        'transformer_model.encoder.embed_tokens.weight',
        'transformer_model.decoder.embed_tokens.weight'
    ]

    for param_name in led_resized_parameter_names:
        # When the loaded state dict belongs to a model that was trained with
        # t5-style denoising pretraining, it has additional parameters that
        # correspond to sentinel tokens. These additional parameters are removed
        # here to ensure that the tensor sizes in the model and the loaded state
        # dict are the same
        if (
                state_dict[param_name].size()
                > token_embedder.state_dict()[param_name].size()
        ):
            target_shape = token_embedder.state_dict()[param_name].size()

            if len(target_shape) == 1:
                new_param = state_dict[param_name][:target_shape[0]]

            elif len(target_shape) == 2:
                new_param = state_dict[param_name][
                    :target_shape[0], :target_shape[1]
                ]
            elif len(target_shape) == 3:
                new_param = state_dict[param_name][
                    :target_shape[0], :target_shape[1], :target_shape[2]
                ]
            else:
                raise NotImplementedError(
                    f'Parameter {param_name} has {len(target_shape)} dimensions, '
                    f'which is more than expected.'
                )

            state_dict[param_name] = new_param

    return state_dict

def prepare_longformer_state_dict_from_structure_benchmark(
        state_dict: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """Prepare a state dict that was saved with the structure benchmark
    with longformer to be loaded into a probing model."""
    state_dict = state_dict['state_dict']

    state_dict = {
        f'transformer_model.{".".join(k.split(".")[1:])}': v
        for k, v in state_dict.items()
        if k.startswith('longformer_model.')
    }

    return state_dict

def prepare_longt5_state_dict_from_structure_benchmark(
        state_dict: Dict[str, Any],
        token_embedder: PretrainedTransformerEmbedder
) -> Dict[str, torch.Tensor]:
    """Prepare a state dict that was saved with the structure benchmark
    with longt5 to be loaded into a probing model."""

    state_dict = state_dict['state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[f'transformer_model.{".".join(k.split(".")[1:])}'] = v

    new_state_dict.pop('transformer_model.lm_head.weight')
    if 'structural_position_embeddings.structural_position_embeddings.weight' \
        in token_embedder.state_dict().keys():
        try:
            new_state_dict['structural_position_embeddings.structural_position_embeddings.weight'] = \
                state_dict['structural_position_embeddings.structural_position_embeddings.weight']
        except KeyError:
            raise KeyError(
                'Probing model has structural position embeddings, '
                'but the loaded state dict does not.'
            )

    # Hard code the names of resized parameters
    longt5_resized_parameter_names = [
        'transformer_model.shared.weight',
        'transformer_model.encoder.embed_tokens.weight',
        'transformer_model.decoder.embed_tokens.weight'
    ]

    for param_name in longt5_resized_parameter_names:
        # When the loaded state dict belongs to a model that was trained with
        # t5-style denoising pretraining, it has additional parameters that
        # correspond to sentinel tokens. These additional parameters are removed
        # here to ensure that the tensor sizes in the model and the loaded state
        # dict are the same
        if (
                new_state_dict[param_name].size()
                > token_embedder.state_dict()[param_name].size()
        ):
            target_shape = token_embedder.state_dict()[param_name].size()

            if len(target_shape) == 1:
                new_param = new_state_dict[param_name][:target_shape[0]]

            elif len(target_shape) == 2:
                new_param = new_state_dict[param_name][
                            :target_shape[0], :target_shape[1]
                            ]
            elif len(target_shape) == 3:
                new_param = new_state_dict[param_name][
                            :target_shape[0], :target_shape[1], :target_shape[2]
                            ]
            else:
                raise NotImplementedError(
                    f'Parameter {param_name} has {len(target_shape)} dimensions, '
                    f'which is more than expected.'
                )

            new_state_dict[param_name] = new_param

    return new_state_dict
