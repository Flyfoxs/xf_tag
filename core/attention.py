import sys
import os

from sklearn.model_selection import StratifiedKFold

from core.callback import *
from tensorflow.python.keras.callbacks import Callback

from core.feature import *
import keras
#max_words = 0


#@lru_cache()

@timed()
def get_train_test(frac=1):

    data = get_data()
    train_data = data.loc[pd.notna(data.type_id)].sample(frac=frac, random_state=2019)
    labels = train_data.type_id.values.tolist()

    test_data =  data.loc[pd.isna(data.type_id)].sample(frac=frac, random_state=2019)

    logger.info(f'Train:{train_data.shape} Test:{test_data.shape}, frac:{frac}')

    feature_col = [col for col in data.columns if col.startswith('fea_') or col.startswith('seq_')]

    label2id, id2label = get_label_id()
    #word2id = get_word2id()

    # Encode input words and labels
    X = train_data.loc[:, feature_col]
    Y = [label2id[label] for label in labels]


    X_test = test_data.loc[:, feature_col]


    return  X, pd.Series(Y, index=train_data.index), X_test


@timed()
def get_model(max_words):
    label2id, id2label = get_label_id()
    word2id = get_word2id()
    manual_features = get_feature_manual(10).shape[1]
    embedding_dim = 100  # The dimension of word embeddings

    # Define input tensor
    sequence_input = keras.Input(shape=(max_words,), dtype='int32')


    word_id_vec = load_embedding(word2vec_tx_mini, type='txt')
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
    model =  load_model(model_path)
    _, _, test = get_train_test()

    label2id, id2label = get_label_id()
    input1_col = [col for col in test.columns if str(col).startswith('seq_')]
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
    return sub_file

def train_base(frac=1):
    X, y, X_test = get_train_test(frac)
    input1_col = [col for col in X.columns if str(col).startswith('seq_')]
    input2_col = [col for col in X.columns if str(col).startswith('fea_')]
    max_words = len(input1_col)
    model = get_model(max_words)

    #get_feature_manual.cache_clear()
    Y_cat = keras.utils.to_categorical(y, num_classes=len(get_app_type_ex()))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
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



class Cal_acc(Callback):

    def __init__(self, val_x, y, X_test):
        super(Cal_acc, self).__init__()
        self.val_x , self.y, self.X_test = val_x, y, X_test

        self.feature_len = self.val_x.shape[1]

        import time, os
        self.batch_id = round(time.time())
        self.model_folder = f'./output/model/{self.batch_id}/'

        os.makedirs(self.model_folder)


        #logger.info(f'Cal_acc base on X:{self.X.shape}, Y:{self.y.shape}')

    @timed()
    def cal_acc(self):
        input1_col = [col for col in self.val_x.columns if str(col).startswith('seq_')]
        input2_col = [col for col in self.val_x.columns if str(col).startswith('fea_')]
        model = self.model
        res = model.predict([self.val_x.loc[:,input1_col], self.val_x.loc[:,input2_col]])

        res = pd.DataFrame(res, index=self.val_x.index)
        acc1, acc2, total = accuracy(res, self.y)
       # logger.info(f'acc1:{acc1:6.5f}, acc2:{acc2:6.5f}, <<<total:{total:6.5f}>>>')

        return acc1, acc2, total


    # def on_train_end(self, logs=None):
    #     acc1, acc2, total = self.cal_acc()
    #     return round(total, 5)

    def on_epoch_end(self, epoch, logs=None):
        acc1, acc2, total = self.cal_acc()
        logger.info(f'Epoch#{epoch}, acc1:{acc1:6.5f}, acc2:{acc2:6.5f}, <<<total:{total:6.5f}>>>')
        if total >= 0.65:
            model_path = f'{self.model_folder}/model_{self.feature_len}_{total:6.5f}_{epoch}.h5'
            self.model.save_model(model_path)
            print(f'weight save to {model_path}')

        threshold = 0.66
        if total >=threshold:
            logger.info(f'Try to gen sub file for local score:{total}, and save to:{model_path}')
            from core.attention import gen_sub
            gen_sub(model_path)
        else:
            logger.info(f'Only gen sub file if the local score >={threshold}, current score:{total}')


        return round(total, 5)



if __name__ == '__main__':
    from fire import Fire
    Fire()

"""
    nohup python -u ./core/attention.py train_base > tf6.log 2>&1 &
    
    #nohup python -u ./core/attention.py train_base   > lda.log 2>&1 &
    
    nohup python -u ./core/attention.py train_base  > cut_all_0.65.log 2>&1 &
    
    
    nohup python -u ./core/attention.py gen_sub ./output/model/1562949408//model_6124_0.66477_2.h5 > gen.log 2>&1 &
    
    
"""
