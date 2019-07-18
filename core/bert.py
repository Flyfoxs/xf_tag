import sys
import os
from multiprocessing import Process

from sklearn.model_selection import StratifiedKFold

from core.callback import *
from tensorflow.python.keras.callbacks import Callback

from core.feature import *
from core.conf import *
import keras
import os


os.environ['TF_KERAS'] = '1'

frac = 0

#Batch size, MAX_len+ex_length, Manual, Manual GP feature cnt, frac
@lru_cache()
@timed()
def get_train_test_bert(frac=1):

    data = get_feature_bert()

    max_bin = 0

    #Keep all the bin group, if it's test data
    data = data.loc[(data.bin<=max_bin) | (pd.isna(data.type_id))]

    timed_bolck('Remove gan data, and len is less then 100')

    data = data.loc[ (data.bin == 0) | (data['len_'] >= 100) ]

    logger.info(f'Train max_bin:{max_bin},Total Bin distribution:\n{data.bin.value_counts().sort_index()}')

    data = data.sort_index()
    logger.info(f'Head of the data:\n, {data.iloc[:3,:3]}')

    train_data = data.loc[pd.notna(data.type_id)].sample(frac=frac, random_state=2019)
    labels = train_data.type_id.values.tolist()
    logger.info(f'Train Bin distribution:\n{train_data.bin.value_counts().sort_index()}')

    test_data =  data.loc[pd.isna(data.type_id)].sample(frac=frac, random_state=2019)
    logger.info(f'Test Bin distribution:\n{test_data.bin.value_counts().sort_index()}')

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

def boost_train(boost=10):
    for _ in range(boost):
        p = Process(target=train_base)
        p.start()
        p.join()




@timed()
def train_base(frac_input=1, fold=0):
    global frac
    frac = frac_input
    BATCH_SIZE = 128
    EPOCHS = 4
    LR = 1e-4

    with timed_bolck(f'Prepare train data#{BATCH_SIZE}'):
        X, y, _ = get_train_test_bert(frac)




        ##Begin to define model
        from keras_bert import load_trained_model_from_checkpoint

        model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, seq_len=SEQ_LEN, )
        #model.summary(line_length=120)

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

    with timed_bolck(f'Training#{fold}'):
        for train_idx, test_idx  in  [list(folds.split(X.values, y))[fold]]:

            logger.info(f'Shape train_x.loc[:, input1_col].iloc[:,0]: {X.loc[:, input1_col].iloc[:,0].shape}')
            train_x, train_y, val_x, val_y = \
                X.iloc[train_idx], Y_cat[train_idx], X.iloc[test_idx], Y_cat[test_idx]

            logger.info(f'get_train_test output: train_x:{train_x.shape}, train_y:{train_y.shape}, val_x:{val_x.shape} ')
            #for sn in range(5):
            input1 = train_x.loc[:, input1_col]#.astype(np.float32)
            input2 = np.zeros_like(input1)#.astype(np.int8)

            logger.info(f'NN Input1:{input1.shape}, Input2:{input2.shape}')

            logger.info(f'NN Input1:{train_x[:3]}')

            from keras_bert import get_custom_objects
            import tensorflow as tf
            with tf.keras.utils.custom_object_scope(get_custom_objects()):
                his = model.fit([input1, input2], train_y,
                                validation_data = ([val_x.loc[:, input1_col], np.zeros_like(val_x.loc[:, input1_col])], val_y),
                                epochs=EPOCHS,  shuffle=True, batch_size=64,
                                callbacks=[Cal_acc(val_x, y.iloc[test_idx], fold )]
                          #steps_per_epoch=1000, validation_steps=10
                          )



            #gen_sub(model, X_test, sn)

    return his

class Cal_acc(Callback):

    def __init__(self, val_x, y , fold):
        super(Cal_acc, self).__init__()
        self.val_x , self.y = val_x, y
        self.fold = fold
        self.feature_len = self.val_x.shape[1]

        self.max_score = 0

        self.score_list = []

        import time, os
        self.batch_id = round(time.time())
        self.model_folder = f'./output/model/{self.batch_id}/'

        os.makedirs(self.model_folder)


        #logger.info(f'Cal_acc base on X:{self.X.shape}, Y:{self.y.shape}')

    #@timed()
    def cal_acc(self):
        input1_col = [col for col in self.val_x.columns if str(col).startswith('bert_')]
        #input2_col = [col for col in self.val_x.columns if str(col).startswith('fea_')]
        #model = self.model
        res = self.model.predict([self.val_x.loc[:,input1_col], np.zeros_like(self.val_x.loc[:,input1_col])])

        res = pd.DataFrame(res, index=self.val_x.index)
        acc1, acc2, total = accuracy(res, self.y)
        res['label'] = self.y
        return acc1, acc2, total, res


    def on_train_end(self, logs=None):
        logger.info(f'Train max:{max(self.score_list)}, at {np.argmax(self.score_list)}/{len(self.score_list)}, Train his:{self.score_list}')

    def on_epoch_end(self, epoch, logs=None):
        print('\n')
        acc1, acc2, total, train = self.cal_acc()

        self.score_list.append(round(total,6))

        # if total >= 0.65:
        #     model_path = f'{self.model_folder}/model_{self.feature_len}_{total:6.5f}_{epoch}.h5'
        #     weight_path = f'{self.model_folder}/weight_{self.feature_len}_{total:6.5f}_{epoch}.h5'
        #
        #     self.model.save_weights(weight_path)
        #     self.model.save(model_path)
        #     print(f'weight save to {model_path}')

        threshold = 0.78
        if total >=threshold and epoch>=1 and total > self.max_score :
            #logger.info(f'Try to gen sub file for local score:{total}, and save to:{model_path}')
            test = self.gen_sub(self.model, f'{self.feature_len}_{total:7.6f}_{epoch}')
            self.save_stack_feature(train, test, f'./output/stacking/{self.fold}_{total:7.6f}.h5')
        else:
            logger.info(f'Only gen sub file if the local score >={threshold}, current score:{total}')

        self.max_score = max(self.max_score, total)

        logger.info(f'Epoch#{epoch}, max:{self.max_score:6.5f}, acc1:{acc1:6.5f}, acc2:{acc2:6.5f}, <<<total:{total:6.5f}>>>')

        print('\n')


        return round(total, 5)

    @staticmethod
    @timed()
    def save_stack_feature(train: pd.DataFrame, test: pd.DataFrame, file_path):
        train.to_hdf(file_path, 'train', mode='a')
        test.to_hdf(file_path, 'test', mode='a')
        logger.info(f'OOF file save to :{file_path}')
        return train, test

    @staticmethod
    @timed()
    #./output/model/1562899782/model_6114_0.65403_2.h5
    def gen_sub(model , info='bert_' , partition_len = 5000):

        global frac
        _, _, test = get_train_test_bert(frac)

        label2id, id2label = get_label_id()
        input1_col = [col for col in test.columns if str(col).startswith('bert_')]
        input2_col = [col for col in test.columns if str(col).startswith('fea_')]

        logger.info(f'Input input1_col:{len(input1_col)}, input2_col:{len(input2_col)}')
        res_list = []
        for sn in tqdm(range(1+ len(test)//partition_len), desc=f'{info}:sub:total:{len(test)},partition_len:{partition_len}'):
            tmp = test.iloc[sn*partition_len: (sn+1)*partition_len]
            #print('\nbegin tmp\n', tmp.iloc[:3,:3].head())
            res = model.predict([ tmp.loc[:,input1_col], np.zeros_like(tmp.loc[:,input1_col]) ])
            res = pd.DataFrame(res, columns=label2id.keys(), index=tmp.index)
            #print('\nend tmp\n', res.iloc[:3, :3].head())
            res_list.append(res)

        res = pd.concat(res_list)
        #print('\nafter concat\n', res.iloc[:3, :3].head())
        res['id'] = res.index
        res.index.name = 'id'
        res['bin'] = res.id.apply(lambda val: int(val.split('_')[1]))
        #print('\nend res\n', res.iloc[:3, :3].head())
        res.to_pickle(f'./output/tmp_sub.pkl')


        res_mean = res.copy(deep=True)
        res_mean['id'] = res_mean.id.apply(lambda val: val.split('_')[0])
        res_select = res_mean.groupby('id')['bin'].agg({'bin_max': 'max'})
        res_select.head()
        res_select = res_select.loc[res_select.bin_max == 3]
        res_mean = res_mean.loc[(res_mean.bin == 0)
                                | ((res_mean.bin == 1) & (res_mean.id.isin(res_select.index)))
                                ]
        logger.info(f'Try to cal avg for res_mean:\n{res_mean.bin.value_counts()}')
        res_mean_len = len(res_mean)
        res_mean = res_mean.groupby('id').mean().sort_index()
        del res_mean['bin']


        res_0 = res.copy(deep=True)
        res_0 = res_0.loc[res_0.bin == 0]
        res_0.index  = res_0.id.apply(lambda val: val.split('_')[0])
        #print('\nres_0\n', res_0.loc[:, ['id', 'bin']].head(3))
        res_0 = res_0.sort_index()
        del res_0['bin']
        del res_0['id']

        for name, res in [('single',res_0), (f'mean_{res_mean_len}', res_mean)]:

            res['label1'] = res.iloc[:, :num_classes].idxmax(axis=1)

            # Exclude top#1
            for index, col in res.label1.items():
                res.loc[index, col] = np.nan

            res['label2'] = res.iloc[:, :num_classes].idxmax(axis=1)


            for col in ['label1','label2']:
                res[col] = res[col].replace(id2label)

            info = info.replace('.','')
            sub_file = f'./output/sub/v18_{info}_{name}.csv'
            res[['label1', 'label2']].to_csv(sub_file)
            logger.info(f'Sub file save to :{sub_file}')

        return res_0

if __name__ == '__main__':
    import fire
    fire.Fire()


"""

nohup python -u ./core/bert.py train_base 0.1  > test.log 2>&1 &
nohup python -u ./core/bert.py train_base  > test.log 2>&1 &
nohup python -u ./core/bert.py train_base  > extend_bert_mean_bin_1.log 2>&1 &

nohup python -u ./core/bert.py boost_train 10 >> boost_1.log 2>&1 &

"""