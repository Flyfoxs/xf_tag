import sys
import os

from sklearn.model_selection import StratifiedKFold

from core.callback import *

from core.feature import *
import keras
#max_words = 0
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

@lru_cache()
@timed()
def get_train_test():
    jieba = get_jieba()
    data = get_data()
    train_data = data.loc[pd.notna(data.type_id)]
    labels = train_data.type_id.values.tolist()

    test_data =  data.loc[pd.isna(data.type_id)]

    logger.info(f'Train:{train_data.shape} Test:{test_data.shape}')


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
    max_words = 0 # maximum number of words in a sentence
    # Construction of word2id dict
    for sentence in input_sentences_train + input_sentences_test:
        if len(sentence) > max_words:
            max_words = len(sentence)
            # logger.debug(f'max_words={max_words}')
    logger.info(f'max_words={max_words}')


    from keras.preprocessing.sequence import pad_sequences

    X = pad_sequences(X, max_words)
    X_test = pad_sequences(X_test, max_words)



    X = pd.DataFrame(X, index=train_data.app_id).add_prefix('word_')
    X_test = pd.DataFrame(X_test, index=test_data).add_prefix('word_')

    data = data.set_index('app_id')
    tfidf_col = [col for col in data.columns if col.startswith('tfidf_')]
    logger.info(f'The tfidf columns:{len(tfidf_col)}, {tfidf_col[:3]} ')

    X_tfidf = data.loc[train_data.app_id, tfidf_col]
    X_test_tfidf = data.loc[test_data.app_id, tfidf_col]

    logger.info(f'Before: X:{X.shape}, X_tfidf:{X_tfidf.shape},X_test:{X_test.shape},  X_test_tfidf:{X_test_tfidf.shape}')

    # Convert Y to numpy array

    X = pd.concat([X, X_tfidf], axis=1)
    X_test = pd.concat([X_test, X_test_tfidf], axis=1)

    logger.info(f'After: X:{X.shape}, X_tfidf:{X_tfidf.shape},X_test:{X_test.shape},  X_test_tfidf:{X_test_tfidf.shape}')


    return  X, pd.Series(Y), X_test


@timed()
def get_model(max_words):
    label2id, id2label = get_label_id()
    word2id = get_word2id()
    tfidf_features = get_tfidf_type().shape[1]
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
    # input_dim = int(lstm_outs.shape[2])
    # permuted_inputs = keras.layers.Permute((2, 1))(lstm_outs)
    attention_vector = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_outs)
    attention_vector = keras.layers.Reshape((max_words,))(attention_vector)
    attention_vector = keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = keras.layers.Dot(axes=1)([lstm_outs, attention_vector])

    # Last layer: fully connected with softmax activation
    fc = keras.layers.Dense(embedding_dim, activation='relu')(attention_output)

    # New input from tfidf
    tfidf_input = keras.Input(shape=(tfidf_features,), dtype='float32')
    dense_tfidf = keras.layers.Dense(len(label2id)*2, activation='relu')(tfidf_input)

    fc_ex = keras.layers.concatenate([fc , dense_tfidf], axis=1)
    output = keras.layers.Dense(len(label2id), activation='softmax')(fc_ex)

    # Finally building model
    model = keras.Model(inputs=[sequence_input, tfidf_input], outputs=output)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')

    # Print model summary
    model.summary()
    return model
@timed()
def gen_sub(model:keras.Model, test:pd.DataFrame, info='0'):
    label2id, id2label = get_label_id()
    input1_col = [col for col in test.columns if not str(col).startswith('tfidf_')]
    input2_col = [col for col in test.columns if str(col).startswith('tfidf_')]

    partition_len = 1000
    res_list = []
    for sn in tqdm(range(1+ len(test)//partition_len), desc=f'total:{len(test)},partition_len:{partition_len}'):
        tmp = test.iloc[sn*partition_len: (sn+1)*partition_len]
        res = model.predict([ tmp.loc[:,input1_col], tmp.loc[:,input2_col] ])
        res = pd.DataFrame(res, columns=label2id.keys(), index=tmp.index)
        res_list.append(res)

    res = pd.concat(res_list)
    id_cnt = res.shape[1]

    res['label1'] = res.iloc[:, :id_cnt].idxmax(axis=1)

    for index, col in res.label1.items():
        res.loc[index, col] = np.nan

    res['label2'] = res.iloc[:, :id_cnt].idxmax(axis=1)


    for col in ['label1','label2']:
        res[col] = res[col].replace(id2label)

    res.index.name = 'id'
    sub_file = f'./output/sub/sub_{info}.csv'
    res[['label1', 'label2']].to_csv(sub_file)
    logger.info(f'Sub file save to :{sub_file}')
    return res

def train_base():
    X, y, X_test = get_train_test()
    input1_col = [col for col in X.columns if not str(col).startswith('tfidf_')]
    input2_col = [col for col in X.columns if str(col).startswith('tfidf_')]
    max_words = len(input1_col)
    model = get_model(max_words)


    Y_cat = keras.utils.to_categorical(y, num_classes=len(get_app_type_ex()))
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
    for train_idx, test_idx  in  folds.split(X.values, y):

        train_x, train_y, val_x, val_y = \
            X.iloc[train_idx], Y_cat[train_idx], X.iloc[test_idx], Y_cat[test_idx]

        logger.info(f'get_train_test output: train_x:{train_x.shape}, train_y:{train_y.shape}, val_x:{val_x.shape}')
        for sn in range(5):
            input1, input2 = train_x.loc[:, input1_col], train_x.loc[:, input2_col]
            logger.info(f'NN Input1:{input1.shape}, Input2:{input2.shape}')
            his = model.fit([input1, input2], train_y,
                            validation_data = ([val_x.loc[:, input1_col], val_x.loc[:, input2_col]], val_y),
                            epochs=8,  shuffle=True, batch_size=64,
                            callbacks=[Cal_acc(val_x, y.iloc[test_idx], X_test)]
                      #steps_per_epoch=1000, validation_steps=10
                      )


            model_path = f'./output/model_{sn}.h5'
            model.save(model_path)
            print(f'weight save to {model_path}')

            gen_sub(model, X_test, sn)

            break



if __name__ == '__main__':
    from fire import Fire
    Fire()

"""
    nohup python -u ./core/attention.py train_base > tf7.log 2>&1 &
"""
