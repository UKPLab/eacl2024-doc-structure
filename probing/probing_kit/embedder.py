from typing import Optional, Dict

from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
import torch
from transformers import LEDModel, LongT5Model
from transformers.models.led.modeling_led import LEDLearnedPositionalEmbedding


class ProbingKitEmbedder(PretrainedTransformerEmbedder):

    def __init__(
            self,
            model_name: str,
            *,
            max_length: int = None,
            train_parameters: bool = True,
            eval_mode: bool = False,
            last_layer_only: bool = True,
            position_embeddings_mode: str = "vanilla",
            batch_injector: Dict = None
    ) -> None:
        # Set training args for LED
        if model_name == 'allenai/led-base-16384':
            transformer_kwargs = {
                'gradient_checkpointing': True,
                'use_cache': False
            }
        else:
            transformer_kwargs = None
        super().__init__(
            model_name,
            max_length=max_length,
            train_parameters=train_parameters,
            eval_mode=eval_mode,
            last_layer_only=last_layer_only,
            transformer_kwargs=transformer_kwargs)

        self.batch_injector = batch_injector
        self.position_embeddings_mode = position_embeddings_mode

        # Prepare position embeddings
        if position_embeddings_mode != "vanilla":
            if isinstance(self.transformer_model, LEDModel):
                led_position_embeddings = self.transformer_model.base_model.encoder.embed_positions

                # noinspection PyTypeChecker
                structural_position_embeddings = LEDAbsoluteStructuralPositionalEmbedding(
                    led_position_embeddings,
                    self.batch_injector,
                    position_embeddings_mode
                )
                self.transformer_model.base_model.encoder.embed_positions = \
                    structural_position_embeddings

            elif isinstance(self.transformer_model, LongT5Model):
                embedding_dim = self.transformer_model.config.d_model
                self.structural_position_embeddings = \
                    LongT5AbsoluteStructuralPositionEmbedding(
                        position_embeddings_mode,
                        embedding_dim
                    )

            else:
                raise NotImplementedError(
                    "Position embeddings mode other than "
                    "vanilla is only supported for LED."
                )

    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """Copy of PretrainedTransformerEmbedder forward method with minor adjustments for LED."""
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings():
                    raise ValueError('Found type ids too large for the chosen transformer model.')
                assert token_ids.shape == type_ids.shape
        fold_long_sequences = self._max_length is not None and token_ids.size(1) > self._max_length
        if fold_long_sequences:
            batch_size, num_segment_concat_wordpieces = token_ids.size()
            token_ids, segment_concat_mask, type_ids = self._fold_long_sequences(
                token_ids, segment_concat_mask, type_ids
            )
        transformer_mask = segment_concat_mask if self._max_length is not None else mask
        assert transformer_mask is not None
        parameters = {
            'input_ids': token_ids,
            'attention_mask': transformer_mask.float(),
            'output_hidden_states': self._scalar_mix is not None
        }
        if not isinstance(self.transformer_model, LongT5Model):
            parameters['global_attention_mask'] = self.batch_injector['global_attention_mask']
        if type_ids is not None:
            parameters['token_type_ids'] = type_ids
        # Add placeholder decoder_output_ids for LED and LongT5
        if isinstance(self.transformer_model, (LEDModel, LongT5Model)):
            parameters['decoder_input_ids'] = torch.zeros((token_ids.shape[0], 1), dtype=torch.long, device=token_ids.device)
        if isinstance(self.transformer_model, LongT5Model):
            # Handle Long T5 separately because tokens need to be embedded separately
            # in order to add structural position embeddings
            input_ids = parameters.pop("input_ids")
            inputs_embeds = self.transformer_model.get_input_embeddings()(input_ids)
            if self.position_embeddings_mode == 'node_types':
                structural_position_ids = self.batch_injector['node_types_ids']
                inputs_embeds += self.structural_position_embeddings(structural_position_ids)
            elif self.position_embeddings_mode == 'node_depths':
                structural_position_ids = self.batch_injector['node_depths_ids']
                inputs_embeds += self.structural_position_embeddings(structural_position_ids)
            parameters['inputs_embeds'] = inputs_embeds
            transformer_output = self.transformer_model(**parameters)
        else:
            transformer_output = self.transformer_model(**parameters)
        if self._scalar_mix is not None:
            # Add placeholder decoder_output_ids for LED and LongT5
            if isinstance(self.transformer_model, (LEDModel, LongT5Model)):
                hidden_states = transformer_output.encoder_hidden_states[1:]
            else:
                hidden_states = transformer_output.hidden_states[1:]
            embeddings = self._scalar_mix(hidden_states)
        else:
            embeddings = transformer_output.last_hidden_state
        if fold_long_sequences:
            embeddings = self._unfold_long_sequences(
                embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces
            )
        return embeddings


###############################################################################
# Added by
###############################################################################

# Adapted from relpos_graph repository (structformer/position_embeddings.py)
class LEDAbsoluteStructuralPositionalEmbedding(torch.nn.Module):
    led_pos_embedding: LEDLearnedPositionalEmbedding

    def __init__(
            self,
            led_pos_embedding: LEDLearnedPositionalEmbedding,
            batch_injector,
            mode: str,
    ) -> None:
        super(LEDAbsoluteStructuralPositionalEmbedding, self).__init__()
        self.led_pos_embedding = led_pos_embedding
        self.batch_injector = batch_injector
        self.mode = mode
        if mode == "node_types":
            num_position_embeddings = 5
        elif mode == "node_depths":
            num_position_embeddings = 21
        else:
            raise ValueError(f"Unknown mode {mode}")

        self.structural_position_embeddings = torch.nn.Embedding(
            num_embeddings=num_position_embeddings,
            embedding_dim=led_pos_embedding.embedding_dim
        )
        # Scale embeddings
        self.structural_position_embeddings.weight.data.normal_(0, 0.0305)

        self.layer_norm = torch.nn.LayerNorm(led_pos_embedding.embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        mode = self.mode
        pos_embeddings = self.led_pos_embedding.forward(input_ids_shape, past_key_values_length)
        pos_embeddings = pos_embeddings.view(1, pos_embeddings.size(0), pos_embeddings.size(1))
        # repeat position embeddings along batch dimension
        pos_embeddings = pos_embeddings.repeat(input_ids_shape[0], 1, 1)

        if mode == "node_types":
            struct_position_ids = self.batch_injector["node_types_ids"]
        elif mode == "node_depths":
            struct_position_ids = self.batch_injector["node_depths_ids"]
        else:
            assert False, f"Unknown position embeddings mode '{mode}'!"

        struct_pos_embeddings = self.structural_position_embeddings(struct_position_ids)

        # pad struct_pos_embeddings to length of pos_embeddings (which is longer to be a multiple of window size)
        padded_struct_pos_embeddings = torch.zeros_like(pos_embeddings)
        padded_struct_pos_embeddings[:, :struct_pos_embeddings.shape[1]] = struct_pos_embeddings

        mixed_pos_embeddings = pos_embeddings + padded_struct_pos_embeddings

        return mixed_pos_embeddings


class LongT5AbsoluteStructuralPositionEmbedding(torch.nn.Module):

    def __init__(
            self,
            mode: str,
            embedding_dim: int,
    ) -> None:
        super(LongT5AbsoluteStructuralPositionEmbedding, self).__init__()
        self.mode = mode
        if mode == "node_types":
            num_position_embeddings = 5
        elif mode == "node_depths":
            num_position_embeddings = 21
        else:
            raise ValueError(f"Unknown mode {mode}")

        self.structural_position_embeddings = torch.nn.Embedding(
            num_embeddings=num_position_embeddings,
            embedding_dim=embedding_dim
        )
        # Scale embeddings
        self.structural_position_embeddings.weight.data.normal_(0, 4.875)

    def forward(self, structural_position_ids: torch.Tensor):

        struct_pos_embeddings = self.structural_position_embeddings(structural_position_ids)

        return struct_pos_embeddings