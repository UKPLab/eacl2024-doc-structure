from __future__ import annotations
from typing import Any, Dict, List, Union

from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


class PreTokenizer(Tokenizer):

    def tokenize(self, tokens: List[Dict[str, Union[str, int]]]) -> List[Token]:
        return [Token(**t) for t in tokens]

    def _to_params(self) -> Dict[str, Any]:
        return {'type': 'pretrained_transformer'}


class DisableSpecialTokens:

    def __init__(self, tokenizer: PretrainedTransformerTokenizer) -> None:
        self._tokenizer = tokenizer
        self._prev = self._tokenizer._add_special_tokens

    def __enter__(self) -> DisableSpecialTokens:
        self._tokenizer._add_special_tokens = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._tokenizer._add_special_tokens = self._prev

    def tokenize(self, text: str) -> List[Token]:
        # Workaround since Transformers special_tokens_mask ignores additional_special_tokens
        tokens = self._tokenizer.tokenize(text)
        return [t for t in tokens if t.text_id not in self._tokenizer.tokenizer.additional_special_tokens_ids]
