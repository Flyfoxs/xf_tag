import sys
import os

from sklearn.model_selection import StratifiedKFold

from core.callback import *
from tensorflow.python.keras.callbacks import Callback

from core.feature import *
from core.conf import *
import keras
import os


os.environ['TF_KERAS'] = '1'






@lru_cache()
@timed()
def get_train_test_bert(frac=1):

    data = get_feature_bert(SEQ_LEN)

    data = data.loc[data.bin<=0]

    train_data = data.loc[pd.notna(data.type_id)].sample(frac=frac, random_state=2019)
    labels = train_data.type_id.values.tolist()

    test_data =  data.loc[pd.isna(data.type_id)].sample(frac=frac, random_state=2019)

    logger.info(f'Train:{train_data.shape} Test:{test_data.shape}, frac:{frac}')

    feature_col = [col for col in data.columns if col.startswith('fea_') or col.startswith('bert_')]

    label2id, id2label = get_label_id()
    #word2id = get_word2id()

    # Encode input words and labels
    X = train_data.loc[:, feature_col]
    Y = [label2id[label] for label in labels]


    X_test = test_data.loc[:, feature_col]


    return  X, pd.Series(Y, index=train_data.index), X_test


# X, y, X_test = get_train_test_bert(0.1)
#
#
# train_x, train_y = load_data(train_path)
# test_x, test_y = load_data(test_path)





def train_base(frac=1):

    X, y, X_test = get_train_test_bert(frac)

    BATCH_SIZE = 128
    EPOCHS = 5
    LR = 1e-4


    ##Begin to define model
    from keras_bert import load_trained_model_from_checkpoint

    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, seq_len=SEQ_LEN, )
    model.summary(line_length=120)

    from tensorflow.python import keras
    from keras_bert import AdamWarmup, calc_train_steps
    inputs = model.inputs[:2]
    dense = model.get_layer('NSP-Dense').output
    outputs = keras.layers.Dense(units=152, activation='softmax')(dense)

    decay_steps, warmup_steps = calc_train_steps(
        y.shape[0],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    model = keras.models.Model(inputs, outputs)
    model.compile(
        AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    ##End to define model

    input1_col = [col for col in X.columns if str(col).startswith('bert_')]
    input2_col = [col for col in X.columns if str(col).startswith('fea_')]
    #max_words = len(input1_col)
    model #= get_model(max_words)

    #get_feature_manual.cache_clear()
    Y_cat = keras.utils.to_categorical(y, num_classes=num_classes)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
    for train_idx, test_idx  in  folds.split(X.values, y):

        logger.info(f'Shape train_x.loc[:, input1_col].iloc[:,0]: {X.loc[:, input1_col].iloc[:,0].shape}')
        train_x, train_y, val_x, val_y = \
            X.iloc[train_idx], Y_cat[train_idx], X.iloc[test_idx], Y_cat[test_idx]

        logger.info(f'get_train_test output: train_x:{train_x.shape}, train_y:{train_y.shape}, val_x:{val_x.shape}, X_test:{X_test.shape}')
        #for sn in range(5):
        input1 = train_x.loc[:, input1_col].astype(np.float32)
        input2 = np.zeros_like(input1).astype(np.int8)

        logger.info(f'NN Input1:{input1.shape}, Input2:{input2.shape}')

        logger.info(f'NN Input1:{val_x[:3]}')
        his = model.fit([input1, input2], train_y,
                        validation_data = ([val_x.loc[:, input1_col], np.zeros_like(val_x.loc[:, input1_col])], val_y),
                        epochs=EPOCHS,  shuffle=True, batch_size=64,
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
        input1_col = [col for col in self.val_x.columns if str(col).startswith('bert_')]
        #input2_col = [col for col in self.val_x.columns if str(col).startswith('fea_')]
        #model = self.model
        res = self.model.predict([self.val_x.loc[:,input1_col], np.zeros_like(self.val_x.loc[:,input1_col])])

        res = pd.DataFrame(res, index=self.val_x.index)
        acc1, acc2, total = accuracy(res, self.y)

        return acc1, acc2, total




    def on_epoch_end(self, epoch, logs=None):
        print('\n')
        acc1, acc2, total = self.cal_acc()
        logger.info(f'Epoch#{epoch}, acc1:{acc1:6.5f}, acc2:{acc2:6.5f}, <<<total:{total:6.5f}>>>')

        # if total >= 0.65:
        #     model_path = f'{self.model_folder}/model_{self.feature_len}_{total:6.5f}_{epoch}.h5'
        #     weight_path = f'{self.model_folder}/weight_{self.feature_len}_{total:6.5f}_{epoch}.h5'
        #
        #     self.model.save_weights(weight_path)
        #     self.model.save(model_path)
        #     print(f'weight save to {model_path}')

        threshold = 0.78
        if total >=threshold:
            #logger.info(f'Try to gen sub file for local score:{total}, and save to:{model_path}')
            gen_sub(self.model, f'{self.feature_len}_{total:6.5f}_{epoch}')
        else:
            logger.info(f'Only gen sub file if the local score >={threshold}, current score:{total}')

        print('\n')
        return round(total, 5)



@timed()
#./output/model/1562899782/model_6114_0.65403_2.h5
def gen_sub(model , info='bert_' , partition_len = 1000):

    _, _, test = get_train_test_bert()

    label2id, id2label = get_label_id()
    input1_col = [col for col in test.columns if str(col).startswith('bert_')]
    input2_col = [col for col in test.columns if str(col).startswith('fea_')]

    logger.info(f'Input input1_col:{len(input1_col)}, input2_col:{len(input2_col)}')
    res_list = []
    for sn in tqdm(range(1+ len(test)//partition_len), desc=f'{info}:sub:total:{len(test)},partition_len:{partition_len}'):
        tmp = test.iloc[sn*partition_len: (sn+1)*partition_len]
        res = model.predict([ tmp.loc[:,input1_col], np.zeros_like(tmp.loc[:,input1_col]) ])
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
    info = info.replace('.','')
    sub_file = f'./output/sub/sub_{info}.csv'
    res[['label1', 'label2']].to_csv(sub_file)
    logger.info(f'Sub file save to :{sub_file}')
    return sub_file

if __name__ == '__main__':
    import fire
    fire.Fire()


"""
nohup python -u ./core/bert.py train_base  > extend_bert_128_bin_1.log 2>&1 &
"""