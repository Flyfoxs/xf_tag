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

#@lru_cache()
@timed()
def get_train_test(frac=1):
    jieba = get_jieba()
    data = get_data()
    train_data = data.loc[pd.notna(data.type_id)].sample(frac=frac, random_state=2019)
    labels = train_data.type_id.values.tolist()

    test_data =  data.loc[pd.isna(data.type_id)].sample(frac=frac, random_state=2019)

    logger.info(f'Train:{train_data.shape} Test:{test_data.shape}, frac:{frac}')

    with timed_bolck('for df cut txt to words'):

        input_sentences_train = [list(jieba.cut(str(text), cut_all=True))  for text in train_data.app_des.values.tolist()]


        input_sentences_test = [list(jieba.cut(str(text), cut_all=True))  for text in test_data.app_des.values.tolist()]


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

    with timed_bolck('X pad_sequences'):
        from keras.preprocessing.sequence import pad_sequences
        X = pad_sequences(X, max_words)
        X_test = pad_sequences(X_test, max_words)



    X      = pd.DataFrame(X, index=train_data.app_id).add_prefix('word_')
    X_test = pd.DataFrame(X_test, index=test_data.app_id).add_prefix('word_')

    data = data.set_index('app_id')
    manual_col = [col for col in data.columns if col.startswith('fea_')]
    #logger.info(f'The tfidf columns:{len(manual_col)}, {manual_col[:3]} ')

    X_manual = data.loc[train_data.app_id, manual_col]
    X_test_manual = data.loc[test_data.app_id, manual_col]

    logger.info(f'Before: X:{X.shape}, X_manual:{X_manual.shape},X_test:{X_test.shape},  X_test_manual:{X_test_manual.shape}')

    # Convert Y to numpy array

    X = pd.concat([X, X_manual], axis=1)
    X_test = pd.concat([X_test, X_test_manual], axis=1)

    logger.info(f'After: X:{X.shape}, X_manual:{X_manual.shape},X_test:{X_test.shape},  X_test_manual:{X_test_manual.shape}')


    return  X, pd.Series(Y), X_test


@timed()
def get_model(max_words):
    label2id, id2label = get_label_id()
    word2id = get_word2id()
    manual_features = get_feature_manual().shape[1]
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

    # New input from manual
    manual_input = keras.Input(shape=(manual_features,), dtype='float32')
    dense_manual = keras.layers.Dense(len(label2id)*2, activation='relu')(manual_input)

    fc_ex = keras.layers.concatenate([fc , dense_manual], axis=1)
    output = keras.layers.Dense(len(label2id), activation='softmax')(fc_ex)

    # Finally building model
    model = keras.Model(inputs=[sequence_input, manual_input], outputs=output)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')

    # Print model summary
    model.summary()
    return model

@timed()
#./output/model/1562899782/model_6114_0.65403_2.h5
def gen_sub(model_path , partition_len = 1000):
    info = model_path.split('/')[-1]
    from keras.models import load_model
    model: keras.Model =  load_model(model_path)
    _, _, test = get_train_test()

    label2id, id2label = get_label_id()
    input1_col = [col for col in test.columns if not str(col).startswith('fea_')]
    input2_col = [col for col in test.columns if str(col).startswith('fea_')]

    logger.info(f'Input input1_col:{len(input1_col)}, input2_col:{len(input2_col)}')
    res_list = []
    for sn in tqdm(range(1+ len(test)//partition_len), desc=f'{info}:sub:total:{len(test)},partition_len:{partition_len}'):
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

def train_base(frac=1):
    X, y, X_test = get_train_test(frac)
    input1_col = [col for col in X.columns if not str(col).startswith('fea_')]
    input2_col = [col for col in X.columns if str(col).startswith('fea_')]
    max_words = len(input1_col)
    model = get_model(max_words)

    get_feature_manual.cache_clear()
    Y_cat = keras.utils.to_categorical(y, num_classes=len(get_app_type_ex()))
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
    for train_idx, test_idx  in  folds.split(X.values, y):

        train_x, train_y, val_x, val_y = \
            X.iloc[train_idx], Y_cat[train_idx], X.iloc[test_idx], Y_cat[test_idx]

        logger.info(f'get_train_test output: train_x:{train_x.shape}, train_y:{train_y.shape}, val_x:{val_x.shape}, X_test:{X_test.shape}')
        for sn in range(5):
            input1, input2 = train_x.loc[:, input1_col], train_x.loc[:, input2_col]
            logger.info(f'NN Input1:{input1.shape}, Input2:{input2.shape}')
            his = model.fit([input1, input2], train_y,
                            validation_data = ([val_x.loc[:, input1_col], val_x.loc[:, input2_col]], val_y),
                            epochs=8,  shuffle=True, batch_size=64,
                            callbacks=[Cal_acc(val_x, y.iloc[test_idx], X_test)]
                      #steps_per_epoch=1000, validation_steps=10
                      )



            #gen_sub(model, X_test, sn)

            break



if __name__ == '__main__':
    from fire import Fire
    Fire()

"""
    nohup python -u ./core/attention.py train_base > tf6.log 2>&1 &
    
    #nohup python -u ./core/attention.py train_base 0.1 > lda.log 2>&1 &
    
    nohup python -u ./core/attention.py train_base  > cut_all_0.65.log 2>&1 &
    
    
    nohup python -u ./core/attention.py gen_sub ./output/model/1562899782/model_6114_0.65403_2.h5 > gen.log 2>&1 &
    
    
"""
