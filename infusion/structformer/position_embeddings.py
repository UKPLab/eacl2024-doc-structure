from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict

import omegaconf
import torch
from torch import nn
if TYPE_CHECKING:
    from transformers.models.led.modeling_led import LEDLearnedPositionalEmbedding

logger = logging.getLogger(__name__)


class LEDAbsoluteStructuralPositionalEmbedding(nn.Module):
    led_pos_embedding: LEDLearnedPositionalEmbedding

    def __init__(self, led_pos_embedding: LEDLearnedPositionalEmbedding, batch_injector, config: omegaconf.DictConfig) -> None:
        super(LEDAbsoluteStructuralPositionalEmbedding, self).__init__()
        self.config = config
        self.led_pos_embedding = led_pos_embedding
        self.batch_injector = batch_injector

        if self.config["position_embeddings"]["mode"] == "node_types":
            num_position_embeddings = len(self.config["node_types"]) + 1  # +1 for prompt/unknown type...
        elif self.config["position_embeddings"]["mode"] == "node_depths":
            num_position_embeddings = self.config["max_depth"] + 1  # +1 for prompt/unknown depth...
        else:
            logger.error(f"Unknown position embeddings mode '{self.config['position_embeddings']['mode']}'!")
            assert False, f"Unknown position embeddings mode '{self.config['position_embeddings']['mode']}'!"

        self.structural_position_embeddings = nn.Embedding(
            num_embeddings=num_position_embeddings,
            embedding_dim=led_pos_embedding.embedding_dim
        )
        # Scale embeddings
        self.structural_position_embeddings.weight.data.normal_(0, self.config['position_embeddings']['init_std'])

        self.layer_norm = nn.LayerNorm(led_pos_embedding.embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        mode = self.config["position_embeddings"]["mode"]
        batch = self.batch_injector["batch"]

        pos_embeddings = self.led_pos_embedding.forward(input_ids_shape, past_key_values_length)
        pos_embeddings = pos_embeddings.view(1, pos_embeddings.size(0), pos_embeddings.size(1))
        # repeat position embeddings along batch dimension
        pos_embeddings = pos_embeddings.repeat(input_ids_shape[0], 1, 1)

        if mode == "node_types":
            struct_position_ids = batch["node_types_labels"]
        elif mode == "node_depths":
            struct_position_ids = batch["node_depths_labels"]
        else:
            logger.error(f"Unknown position embeddings mode '{mode}'!")
            assert False, f"Unknown position embeddings mode '{mode}'!"

        struct_pos_embeddings = self.structural_position_embeddings(struct_position_ids)

        # pad struct_pos_embeddings to length of pos_embeddings (which is longer to be a multiple of window size)
        padded_struct_pos_embeddings = torch.zeros_like(pos_embeddings)
        padded_struct_pos_embeddings[:, :struct_pos_embeddings.shape[1]] = struct_pos_embeddings 

        mixed_pos_embeddings = pos_embeddings + padded_struct_pos_embeddings

        return mixed_pos_embeddings


class LongformerAbsoluteStructuralPositionEmbedding(nn.Module):
    longformer_pos_embedding: nn.Module

    def __init__(self, longformer_pos_embedding: nn.Module, batch_injector, config: omegaconf.DictConfig) -> None:
        super(LongformerAbsoluteStructuralPositionEmbedding, self).__init__()
        self.config = config
        self.longformer_pos_embedding = longformer_pos_embedding
        self.batch_injector = batch_injector

        num_position_embeddings = get_num_position_embeddings(
            self.config['position_embeddings']['mode'],
            len(self.config['input_sequence']['include_node_types']),
            self.config['max_depth']
        )

        self.structural_position_embeddings = nn.Embedding(
            num_embeddings=num_position_embeddings,
            embedding_dim=longformer_pos_embedding.embedding_dim
        )

        # Scale embeddings
        self.structural_position_embeddings.weight.data.normal_(0, self.config['position_embeddings']['init_std'])

        self.layer_norm = nn.LayerNorm(longformer_pos_embedding.embedding_dim)

    def forward(self, positions: torch.Tensor):
        mode = self.config["position_embeddings"]["mode"]
        batch = self.batch_injector["batch"]

        pos_embeddings = self.longformer_pos_embedding(positions)

        if mode == "node_types":
            struct_position_ids = batch["node_types_labels"]
        elif mode == "node_depths":
            struct_position_ids = batch["node_depths_labels"]
        else:
            logger.error(f"Unknown position embeddings mode '{mode}'!")
            assert False, f"Unknown position embeddings mode '{mode}'!"

        struct_pos_embeddings = self.structural_position_embeddings(struct_position_ids)

        # pad struct_pos_embeddings to length of pos_embeddings (which is longer to be a multiple of window size)
        padded_struct_pos_embeddings = torch.zeros_like(pos_embeddings)
        padded_struct_pos_embeddings[0, :struct_pos_embeddings.shape[1]] = struct_pos_embeddings[0]  # TODO: only works for batch size 1

        mixed_pos_embeddings = pos_embeddings + padded_struct_pos_embeddings

        return mixed_pos_embeddings


class LEDLearnedEmbedding(nn.Embedding):
    """
    This module learns embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float):
        super().__init__(num_embeddings, embedding_dim)
        # Scale embeddings
        self.weight.data.normal_(0, init_std)

    def forward(self, embed_ids: torch.Tensor):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        return super().forward(embed_ids)


def get_num_position_embeddings(
        position_embeddings_mode: str,
        n_node_types: int = None,
        max_depth: int = None
) -> int:
    """
    Get the number of embeddings needed
    """
    if position_embeddings_mode == 'vanilla':
        num_position_embeddings = 0
    elif position_embeddings_mode == "node_types":
        num_position_embeddings = n_node_types + 1  # +1 for prompt/unknown type...
    elif position_embeddings_mode == "node_depths":
        num_position_embeddings = max_depth + 1  # +1 for prompt/unknown depth...
    else:
        logger.error(f"Unknown position embeddings mode '{position_embeddings_mode}'!")
        assert False, f"Unknown position embeddings mode '{position_embeddings_mode}'!"

    return num_position_embeddings


def get_post_encoder_position_embedding_ids(
        post_encoder_position_embeddings_mode: str,
        tokenized_structure: Dict[str, torch.Tensor]
) -> torch.Tensor:
    if post_encoder_position_embeddings_mode == 'vanilla':
        post_encoder_position_embedding_ids = None
    elif post_encoder_position_embeddings_mode == "node_types":
        post_encoder_position_embedding_ids = tokenized_structure['node_types_labels']
    elif post_encoder_position_embeddings_mode == "node_depths":
        post_encoder_position_embedding_ids = tokenized_structure['node_depths_labels']
    else:
        logger.error(f"Unknown position embeddings mode '{post_encoder_position_embeddings_mode}'!")
        assert False, f"Unknown position embeddings mode '{post_encoder_position_embeddings_mode}'!"

    return post_encoder_position_embedding_ids


class LongT5StructuralPositionEmbedding(nn.Module):

    def __init__(
            self,
            config: omegaconf.DictConfig,
            hidden_dim: int
    ) -> None:
        super(LongT5StructuralPositionEmbedding, self).__init__()
        self.config = config

        num_position_embeddings = get_num_position_embeddings(
            self.config['position_embeddings']['mode'],
            len(self.config['input_sequence']['include_node_types']),
            self.config['max_depth']
        )

        self.structural_position_embeddings = nn.Embedding(
            num_embeddings=num_position_embeddings,
            embedding_dim=hidden_dim
        )

        # Scale embeddings
        self.structural_position_embeddings.weight.data.normal_(0, self.config['position_embeddings']['init_std'])

    def forward(self, structural_position_ids: torch.Tensor):

        struct_pos_embeddings = self.structural_position_embeddings(structural_position_ids)

        return struct_pos_embeddings