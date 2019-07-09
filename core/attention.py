import sys
import os

from core.feature import *
import keras
max_words = 0
word2vec_tx, vector_size = './input/mini_tx.kv',  200

@lru_cache()
def get_word2id():

    word_id_vec =  load_embedding(word2vec_tx, type='txt')

    word2id = {l: i for i, l in enumerate(set(word_id_vec.index.values))}

    return word2id


@lru_cache()
def get_label_id():
    app_type  = get_app_type_ex()
    labels = app_type.type_id.values.tolist()
    # Construction of label2id and id2label dicts
    label2id = {l: i for i, l in enumerate(set(labels))}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label

@timed()
def get_train_test():
    jieba = get_jieba()
    data = get_data()
    train_data = data.loc[pd.notna(data.type_id)]
    labels = train_data.type_id.values.tolist()

    test_data =  data.loc[pd.isna(data.type_id)]


    input_sentences_train = [list(jieba.cut(str(text), cut_all=False))  for text in train_data.app_des.values.tolist()]


    input_sentences_test = [list(jieba.cut(str(text), cut_all=False))  for text in test_data.app_des.values.tolist()]


    # word2vec_tx, vector_size = './input/mini_tx.kv',  200
    #
    #
    #
    # word2id = {l: i for i, l in enumerate(set(word_id_vec.index.values))}
    #word2id = dict( word_id_vec.reset_index().apply(lambda row: (row['word'], row.index), axis=1).values )
    #logger.debug(f'Word length:{len(word2id)}')


    #embedding_weights[10]

    # Initialize word2id and label2id dictionaries that will be used to encode words and labels


    # id2label

    label2id, id2label = get_label_id()
    word2id = get_word2id()

    # Encode input words and labels
    X = [[word2id[word] for word in sentence if word in word2id] for sentence in input_sentences_train]
    Y = [label2id[label] for label in labels]


    X_test = [[word2id[word] for word in sentence if word in word2id] for sentence in input_sentences_test]

    # Apply Padding to X
    global max_words  # maximum number of words in a sentence
    # Construction of word2id dict
    for sentence in input_sentences_train + input_sentences_test:
        if len(sentence) > max_words:
            max_words = len(sentence)
            # logger.debug(f'max_words={max_words}')
    logger.info(f'max_words={max_words}')


    from keras.preprocessing.sequence import pad_sequences

    X = pad_sequences(X, max_words)
    X_test = pad_sequences(X_test, max_words)




    # Convert Y to numpy array
    Y = keras.utils.to_categorical(Y, num_classes=len(label2id))

    # Print shapes
    print("Shape of X: {}".format(X.shape))
    print("Shape of Y: {}".format(Y.shape))

    return X, Y, X_test


@timed()
def get_model(max_words):
    word2id = get_word2id()
    embedding_dim = 100  # The dimension of word embeddings

    # Define input tensor
    sequence_input = keras.Input(shape=(max_words,), dtype='int32')

    word_id_vec = load_embedding(word2vec_tx, type='txt')
    embedding_weights = word_id_vec.iloc[:, -vector_size:].fillna(0).values
    # Word embedding layer

    logger.info(f'Mode Paras:embedding_dim:{embedding_dim}, word2id:{len(word2id)}, max_words:{max_words},vector_size:{vector_size}, embedding_weights:{embedding_weights.shape}')
    embedded_inputs = keras.layers.Embedding(len(word2id),
                                             vector_size,
                                             input_length=max_words,
                                             weights=[embedding_weights],
                                             )(sequence_input)

    # Apply dropout to prevent overfitting
    embedded_inputs = keras.layers.Dropout(0.2)(embedded_inputs)

    # Apply Bidirectional LSTM over embedded inputs
    lstm_outs = keras.layers.wrappers.Bidirectional(
        keras.layers.LSTM(embedding_dim, return_sequences=True)
    )(embedded_inputs)

    # Apply dropout to LSTM outputs to prevent overfitting
    lstm_outs = keras.layers.Dropout(0.2)(lstm_outs)

    # Attention Mechanism - Generate attention vectors
    input_dim = int(lstm_outs.shape[2])
    permuted_inputs = keras.layers.Permute((2, 1))(lstm_outs)
    attention_vector = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_outs)
    attention_vector = keras.layers.Reshape((max_words,))(attention_vector)
    attention_vector = keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = keras.layers.Dot(axes=1)([lstm_outs, attention_vector])

    # Last layer: fully connected with softmax activation
    fc = keras.layers.Dense(embedding_dim, activation='relu')(attention_output)

    label2id, id2label = get_label_id()
    output = keras.layers.Dense(len(label2id), activation='softmax')(fc)

    # Finally building model
    model = keras.Model(inputs=[sequence_input], outputs=output)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')

    # Print model summary
    model.summary()
    return model

def gen_sub(model:keras.Model, test:pd.DataFrame, sn=0):
    res = model.predict(test)
    label2id, id2label = get_label_id()
    res = pd.DataFrame(res, columns=label2id.keys(), index=test.index)

    id_cnt = res.shape[1]

    res['label1'] = res.iloc[:, :id_cnt].idxmax(axis=1)

    for index, col in res.label1.items():
        res.loc[index, col] = np.nan

    res['label2'] = res.iloc[:, :id_cnt].idxmax(axis=1)


    for col in ['label1','label2']:
        res[col] = res[col].replace(id2label)

    res.index.name = 'id'
    res[['label1', 'label2']].to_csv(f'./output/sub/sub_{sn}.csv')

def train_base():
    X, Y, X_test = get_train_test()
    max_words = X.shape[1]
    model = get_model(max_words)

    logger.info(f'get_train_test output: X:{X.shape}, Y:{Y.shape}, X_test:{X_test.shape}')
    for sn in range(5):
        model.fit(X, Y, epochs=2, batch_size=64, validation_split=0.1, shuffle=True)
        gen_sub(model, X_test, sn)



if __name__ == '__main__':
    from fire import Fire
    Fire()

"""
    nohup python -u ./core/attention.py train_base > att.log 2>&1 &
"""
