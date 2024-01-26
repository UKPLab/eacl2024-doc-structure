from __future__ import annotations

import copy
import dataclasses
import random
from collections import UserDict
from typing import Union, Dict, List, Tuple, Optional

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase, CharSpan
from tokenizers import Encoding
import numpy as np


"""Large pieces of code copied from 
https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py"""


class SimpleBatchEncoding(UserDict):
    def __init__(
            self,
            data: Dict[str, Union[List, np.ndarray, torch.Tensor]] = None,
            encoding: List[Union[SimpleEncoding, Encoding]] = None
    ):
        super().__init__(data)
        self.encodings: List[SimpleEncoding] = []
        if encoding is not None:
            self.encodings = [
                SimpleEncoding.from_encoding(encoding_)
                for encoding_ in encoding
            ]

    def __getitem__(self, item):
        if type(item) is str:
            return self.data[item]
        elif type(item) is int:
            return self.encodings[item]
        else:
            raise ValueError(f'item should be str or int but got {type(item)}')

    def __setitem__(self, key, value):
        self.data[key] = value

    def token_to_chars(self, batch_or_token_index: int, token_index: Optional[int] = None) -> CharSpan:
        """
        Get the character span corresponding to an encoded token in a sequence of the batch.

        Character spans are returned as a [`~tokenization_utils_base.CharSpan`] with:

        - **start** -- Index of the first character in the original string associated to the token.
        - **end** -- Index of the character following the last character in the original string associated to the
          token.

        Can be called as:

        - `self.token_to_chars(token_index)` if batch size is 1
        - `self.token_to_chars(batch_index, token_index)` if batch size is greater or equal to 1

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token or tokens in
                the sequence.

        Returns:
            [`~tokenization_utils_base.CharSpan`]: Span of characters in the original string.
        """

        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        return CharSpan(*(self.encodings[batch_index].token_to_chars(token_index)))

    def char_to_token(
        self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0
    ) -> int:
        """
        Get the index of the token in the encoded output comprising a character in the original string for a sequence
        of the batch.

        Can be called as:

        - `self.char_to_token(char_index)` if batch size is 1
        - `self.char_to_token(batch_index, char_index)` if batch size is greater or equal to 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            char_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            `int`: Index of the token.
        """

        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self.encodings[batch_index].char_to_token(char_index, sequence_index)


    @classmethod
    def from_batch_encoding(cls, batch_encoding: Union[SimpleBatchEncoding, BatchEncoding]):
        return cls(batch_encoding.data, batch_encoding.encodings)

    def data_to_torch(self):
        for key, value in self.data.items():
            try:
                new_value = torch.tensor(value)
            except (TypeError, RuntimeError):
                # Fails when value is not set (None)
                new_value = value
            self.data[key] = new_value


@dataclasses.dataclass
class SimpleEncoding:
    attention_mask: List[int]
    ids: List[int]
    n_sequences: int
    offsets: List[Tuple[int, int]]
    overflowing: List
    sequence_ids: List[int]
    special_tokens_mask: List[int]
    tokens: List[str]
    type_ids: List[int]
    word_ids: List[int]
    words: List[int]

    @classmethod
    def from_encoding(cls, encoding: Encoding):
        return cls(
            encoding.attention_mask,
            encoding.ids,
            encoding.n_sequences,
            encoding.offsets,
            encoding.overflowing,
            encoding.sequence_ids,
            encoding.special_tokens_mask,
            encoding.tokens,
            encoding.type_ids,
            encoding.word_ids,
            encoding.words
        )

    def char_to_token(self, char_pos, sequence_index=0):
        token_index = None
        for i, (offsets, sequence_id) in enumerate(zip(self.offsets, self.sequence_ids)):
            if sequence_id == sequence_index:
                if char_pos in range(*offsets):
                    token_index = i

        return token_index

    def token_to_chars(self, token_index):
        return self.offsets[token_index]

    @staticmethod
    def merge(encodings: List[SimpleEncoding], growing_offsets=True):
        if growing_offsets:
            raise NotImplementedError
        first_encoding = encodings.pop(0)

        if type(first_encoding) is Encoding:
            first_encoding = SimpleEncoding.from_encoding(first_encoding)

        for encoding in encodings:
            for field in dataclasses.fields(SimpleEncoding):
                if not field.name == 'n_sequences':
                    getattr(first_encoding, field.name).extend(getattr(encoding, field.name))

        first_encoding.n_sequences = len(set(first_encoding.sequence_ids))

        return first_encoding

    def truncate(self, max_length, stride=0, direction='right'):
        if stride != 0:
            raise NotImplementedError

        if direction == 'right':
            for field in dataclasses.fields(self):
                if not field.name == 'n_sequences':
                    setattr(self, field.name, getattr(self, field.name)[:max_length])

        elif direction == 'left':
            for field in dataclasses.fields(self):
                if not field.name == 'n_sequences':
                    setattr(self, field.name, getattr(self, field.name)[-max_length:])

        else:
            raise NotImplementedError

    def empty_overflowing(self):
        self.overflowing = []

    @classmethod
    def create_empty_instance(cls):
        return cls(
            [],
            [],
            0,
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            []
        )

def get_noise_density(
        input_length: int,
        desired_output_length: int,
        mean_noise_span_length: int
):
    # + 1 because of sentinel token
    n_noise_spans = desired_output_length / (mean_noise_span_length + 1)

    noise_density = (mean_noise_span_length * n_noise_spans) / input_length

    return noise_density


def get_n_noise_spans(
        input_length: int,
        noise_density: float,
        mean_noise_span_length: Union[float, int]
) -> int:
    num_noise_tokens = int(np.round(input_length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), input_length)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))
    return num_noise_spans


def t5_mask_input_and_get_output(
        batch: Union[SimpleBatchEncoding, BatchEncoding],
        noise_density: float,
        mean_noise_span_length: Union[float, int],
        sentinel_start_id: int,
        eos_token_id: int,
        input_length: int,
        target_length: int,
        pad_token_id: int,
        decoder_start_token_id: int,
        tokenizer: PreTrainedTokenizerBase,
        dynamic_noise_span_length: bool = False
) -> SimpleBatchEncoding:
    input_ids = batch["input_ids"]
    batch_size, expanded_input_length = input_ids.shape

    new_batch = SimpleBatchEncoding.from_batch_encoding(batch)
    del batch
    batch = new_batch

    for encoding in batch.encodings:
        encoding.empty_overflowing()

    if dynamic_noise_span_length:
        mean_noise_span_length = random.choice([4, 8, 12])

    # mask is similar for all sequences in batch
    mask_indices = np.asarray([
        random_spans_noise_mask(expanded_input_length, noise_density, mean_noise_span_length)
        for _ in range(batch_size)
    ])
    labels_mask = ~mask_indices

    input_ids_sentinel = create_sentinel_ids(
        mask_indices.astype(np.int8),
        False
    )
    labels_sentinel = create_sentinel_ids(
        labels_mask.astype(np.int8),
        True,
        sentinel_start_id
    )

    batch = concatenate_input(
        batch,
        input_ids_sentinel,
        tokenizer
    )
    batch["labels"] = filter_input_ids(
        input_ids,
        labels_sentinel,
        eos_token_id
    )

    # if batch["input_ids"].shape[-1] != input_length:
    #     raise ValueError(
    #         f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
    #         f" should be {input_length}."
    #     )
    #
    # if batch["labels"].shape[-1] != target_length:
    #     raise ValueError(
    #         f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
    #         f" {target_length}."
    #     )

    # to check that tokens are correctly preprocessed, one can run
    # `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
    try:
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], pad_token_id, decoder_start_token_id
        )
    except TypeError:
        # shift_tokens_right is not implemented for all tokenizers
        batch['decoder_input_ids'] = None


    return batch


def create_sentinel_ids(
        mask_indices: np.ndarray,
        add_sentinel_start_id: bool,
        sentinel_start_id: int = None
):
    """
    Sentinel ids creation given the indices that should be masked.
    The start indices of each mask are replaced by the sentinel ids in increasing
    order. Consecutive mask indices to be deleted are replaced with `-1`.
    """
    start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
    start_indices[:, 0] = mask_indices[:, 0]

    sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
    if add_sentinel_start_id:
        sentinel_ids = np.where(sentinel_ids != 0, (sentinel_start_id - 1 + sentinel_ids), 0)
    sentinel_ids -= mask_indices - start_indices

    return sentinel_ids


def concatenate_input(inputs: SimpleBatchEncoding, sentinel_ids: np.ndarray, tokenizer: PreTrainedTokenizerBase):

    def make_sentinel_encoding(sentinel_token):
        sentinel_encoding = tokenizer(
            sentinel_token,
            add_special_tokens=False
        ).encodings[0]
        sentinel_simple_encoding = SimpleEncoding.from_encoding(sentinel_encoding)
        sentinel_simple_encoding.sequence_ids = [1]
        return sentinel_simple_encoding

    def concatenate_single_encoding(
            encoding: SimpleEncoding,
            sentinel_ids_for_encoding
    ) -> SimpleEncoding:

        concatenated_simple_encoding = SimpleEncoding.create_empty_instance()

        i = 0
        start_idx = 0
        # Go over sentinel ids, find the segments that are non-masked and those that are masked
        # Merge the non-masked segments to the masked segments of length 1, keep
        # the character offsets
        while i < len(encoding.ids):
            if sentinel_ids_for_encoding[i] == 0:
                if start_idx is None:
                    start_idx = i
            elif sentinel_ids_for_encoding[i] != -1:
                sentinel_id = sentinel_ids_for_encoding[i]
                # Make copy to keep original encoding intact
                piece = copy.deepcopy(encoding)
                # Cut off right
                piece.truncate(i)
                # Cut off left
                piece.truncate(i - start_idx, direction='left')

                sentinel_token_encoding = make_sentinel_encoding(
                    f'<sentinel_{sentinel_id - 1}>'  # -1 because ids start at 1, but tokens start at 0
                )
                concatenated_simple_encoding = concatenated_simple_encoding.merge(
                    [concatenated_simple_encoding, piece, sentinel_token_encoding],
                    growing_offsets=False
                )

                del piece
                del sentinel_token_encoding

                start_idx = None

            i += 1

        # Add final part
        if start_idx is not None:
            if start_idx < len(encoding.ids):
                piece = copy.deepcopy(encoding)
                # Cut off left
                piece.truncate(i - start_idx)
                # Merge
                concatenated_simple_encoding = concatenated_simple_encoding.merge(
                    [concatenated_simple_encoding, piece],
                    growing_offsets=False
                )

        return concatenated_simple_encoding

    # concatenate individual encodings according to sentinel ids
    new_encodings = [
        concatenate_single_encoding(encoding, sentinel_ids_for_encoding_)
        for encoding, sentinel_ids_for_encoding_ in zip(inputs.encodings, sentinel_ids)
    ]

    # Combine encodings into single BatchEncoding
    # Padding should not be needed
    input_ids = np.stack(
        [
            encoding.ids
            for encoding in new_encodings
        ],
    )
    attention_mask = np.stack(
        [
            encoding.attention_mask
            for encoding in new_encodings
        ]
    )
    result = SimpleBatchEncoding(
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        },
        encoding=new_encodings
    )

    return result


def filter_input_ids(input_ids, sentinel_ids, eos_token_id):
    """
    Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
    This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
    """
    batch_size = input_ids.shape[0]

    input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
    # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
    # masked tokens coming after sentinel tokens and should be removed
    input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
    input_ids = np.concatenate(
        [input_ids, np.full((batch_size, 1), eos_token_id, dtype=np.int32)], axis=-1
    )
    return input_ids


def random_spans_noise_mask(
        length: int,
        noise_density: float,
        mean_noise_span_length: Union[int, float]
):
    """This function is copy of
    `random_spans_helper
    <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]


# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


def random_spans_helper(
        inputs_length: int,
        noise_density: float,
        mean_noise_span_length: int,
        extra_tokens_per_span_inputs: int,
        extra_tokens_per_span_targets: int,
        verbose=False
):
  """
  Copied from https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466
  Training parameters to avoid padding with random_spans_noise_mask.
  When training a model with random_spans_noise_mask, we would like to set the
  other training hyperparmeters in a way that avoids padding.  This function
  helps us compute these hyperparameters.
  We assume that each noise span in the input is replaced by
  extra_tokens_per_span_inputs sentinel tokens, and each non-noise span in the
  targets is replaced by extra_tokens_per_span_targets sentinel tokens.
  This function tells us the required number of tokens in the raw example (for
  split_tokens()) as well as the length of the encoded targets.
  Note that this function assumes the inputs and targets will have EOS appended
  and includes that in the reported length.
  Args:
    inputs_length: an integer - desired length of the tokenized inputs sequence
    noise_density: a float
    mean_noise_span_length: a float
    extra_tokens_per_span_inputs: an integer
    extra_tokens_per_span_targets: an integer
    verbose: a bool indicating whether to log sequence lengths
  Returns:
    tokens_length: length of original text in tokens
    targets_length: an integer - length in tokens of encoded targets sequence
  """
  def _tokens_length_to_inputs_length_targets_length(tokens_length):
    num_noise_tokens = int(round(tokens_length * noise_density))
    num_nonnoise_tokens = tokens_length - num_noise_tokens
    num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
    # inputs contain all nonnoise tokens, sentinels for all noise spans
    # and one EOS token.
    return (
        num_nonnoise_tokens +
        num_noise_spans * extra_tokens_per_span_inputs + 1,
        num_noise_tokens +
        num_noise_spans * extra_tokens_per_span_targets + 1)

  tokens_length = inputs_length
  while (_tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
         <= inputs_length):
    tokens_length += 1
  inputs_length, targets_length = (
      _tokens_length_to_inputs_length_targets_length(tokens_length))
  # minor hack to get the targets length to be equal to inputs length
  # which is more likely to have been set to a nice round number.
  if noise_density == 0.5 and targets_length > inputs_length:
    tokens_length -= 1
    targets_length -= 1
  if verbose:
    logging.info(
        'tokens_length=%s inputs_length=%s targets_length=%s '
        'noise_density=%s mean_noise_span_length=%s ',
        tokens_length, inputs_length, targets_length,
        noise_density, mean_noise_span_length)
  return tokens_length, targets_length


if __name__ == '__main__':
    import transformers

    NOISE_DENSITY = 0.25
    MEAN_NOISE_SPAN_LENGTH = 2

    tokenizer_ = transformers.AutoTokenizer.from_pretrained(
        'allenai/longformer-base-4096',
        local_files_only=True,
    )
    sentinel_start_id_ = len(tokenizer_.vocab)

    sentinel_tokens = [
        f'<sentinel_{i}>'
        for i in range(200)
    ]

    tokenizer_.add_tokens(sentinel_tokens)

    text = 40 * 'I like pizza. Do you like pizza, too? I hope you do, because it is excellent food. '

    batch_encoding = tokenizer_(
        text,
        return_tensors='pt',
        add_special_tokens=False
    )

    input_length_, target_length_ = compute_input_and_target_lengths(
        len(batch_encoding['input_ids'][0]),
        NOISE_DENSITY,
        MEAN_NOISE_SPAN_LENGTH
    )

    t5_mask_input_and_get_output(
        batch_encoding,
        NOISE_DENSITY,
        MEAN_NOISE_SPAN_LENGTH,
        sentinel_start_id_,
        tokenizer_.eos_token_id,
        input_length_,
        target_length_,
        tokenizer_.pad_token_id,
        tokenizer_.bos_token_id,
        tokenizer_
    )
    