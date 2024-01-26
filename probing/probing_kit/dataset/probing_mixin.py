from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from intertext_graph.itgraph import IntertextDocument


class ProbingMixin:

    _tokenizer: PretrainedTransformerTokenizer
    _max_length: int
    _allow_list = ['article-title', 'abstract', 'title', 'p']

    def _token_filter(self, doc: IntertextDocument) -> bool:
        """Filter docs for tokens [1200, max_length]."""
        # Previous filter did not know about structure tokens
        return 1200 <= len(self._tokenizer.tokenize(doc.to_plaintext(self._allow_list))) <= self._max_length
