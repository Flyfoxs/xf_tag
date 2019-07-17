from keras_bert import Tokenizer

class Tokenizer(Tokenizer):

    def convert_tokens_to_ids(self, tokens):
        tokens = self._tokenize(tokens)
        token_ids = self._convert_tokens_to_ids(tokens)
        return token_ids

