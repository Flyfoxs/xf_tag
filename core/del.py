from keras_bert import Tokenizer

token_dict = {
    '[CLS]': 0,
    '[SEP]': 1,
    'un': 2,
    '##aff': 3,
    '##able': 4,
    '[UNK]': 5,
}
tokenizer = Tokenizer(token_dict)
print(tokenizer.tokenize('unaffable'))  # The result should be `['[CLS]', 'un', '##aff', '##able', '[SEP]']`
indices, segments = tokenizer.encode('unaffable')
print(indices)  # Should be `[0, 2, 3, 4, 1]`
print(segments)  # Should be `[0, 0, 0, 0, 0]`
