import sys
import os
from multiprocessing import Process

from sklearn.model_selection import StratifiedKFold


from tensorflow.python.keras.callbacks import Callback

from core.feature import *
from core.conf import *
import keras
import os


os.environ['TF_KERAS'] = '1'


#Batch size, MAX_len+ex_length, Manual, Manual GP feature cnt, frac
@lru_cache()
@timed()
def get_train_test_bert():

    frac = get_args().frac
    max_bin = get_args().max_bin

    data = get_feature_bert()

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
def train_base(args):
    #frac = args.frac
    fold = args.fold
    EPOCHS = args.epochs

    BATCH_SIZE = 128
    LR = 1e-4

    with timed_bolck(f'Prepare train data#{BATCH_SIZE}'):
        X, y, _ = get_train_test_bert()




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
        #folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

    with timed_bolck(f'Training#{fold}'):
        from core.split import split_df_by_index
        train_idx, test_idx = split_df_by_index(pd.Series(X.index),fold)

        logger.info(f'Shape train_x.loc[:, input1_col].iloc[:,0]: {X.loc[:, input1_col].iloc[:,0].shape}')
        train_x, train_y, val_x, val_y = \
            X.iloc[train_idx], Y_cat[train_idx], X.iloc[test_idx], Y_cat[test_idx]

        logger.info(f'get_train_test output: train_x:{train_x.shape}, train_y:{train_y.shape}, val_x:{val_x.shape} ')
        #for sn in range(5):
        input1 = train_x.loc[:, input1_col]#.astype(np.float32)
        input2 = np.zeros_like(input1)#.astype(np.int8)

        logger.info(f'NN Input1:{input1.shape}, Input2:{input2.shape}')

        logger.info(f'NN train_x:{train_x[:3]}')

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
        self.gen_file = False

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
        val = self.model.predict([self.val_x.loc[:,input1_col], np.zeros_like(self.val_x.loc[:,input1_col])])

        label2id, id2label = get_label_id()
        val = pd.DataFrame(val, columns=label2id.keys(), index=self.val_x.index)
        val['label'] = self.y.astype(int).replace(id2label).astype(int)
        val['bin'] = pd.Series(val.index).str[-1].values.astype(int)
        logger.info(f'Head val#label:\n{val.label.head()}')
        res_val = val.copy()
        # res_val.to_pickle(f'./output/tmp_res_val.pkl')
        # logger.info(f'Debug file: save to ./output/tmp_res_val.pkl')

        logger.info(f'val bin:\n {val.bin.value_counts()}')
        #y2['bin'] = pd.Series(y2.index).str[-1].values.astype(int)
        for bin_list in [[0,1], [0]]:
            tmp_val = val.loc[val.bin.isin(bin_list)].copy()
            if len(tmp_val)>0:

                tmp_val['app_id'] = pd.Series(tmp_val.index).apply(lambda val: val.split('_')[0]).values
                df_len = len(tmp_val)
                tmp_val = tmp_val.drop_duplicates(['app_id', 'bin'])

                tmp_val_mean = tmp_val.groupby('app_id').mean()
                tmp_val_max = tmp_val.groupby('app_id').max()
                for name, df in [('mean', tmp_val_mean), ('max', tmp_val_max)]:
                    acc1, acc2, total = accuracy(df)
                    logger.info(f'Val({name})#{len(df)}/{df_len}, bin#{bin_list}, acc1:{acc1}, acc2:{acc2}, total:<<<{total}>>>')
            else:
                logger.info(f'Can not find bin:{bin_list} in val')
        return acc1, acc2, total, res_val


    def on_train_end(self, logs=None):
        logger.info(f'Train max:{max(self.score_list)}, at {np.argmax(self.score_list)}/{len(self.score_list)-1}, Train his:{self.score_list}, gen_file:{self.gen_file}')

    def on_epoch_end(self, epoch, logs=None):
        print('\n')
        acc1, acc2, total, val = self.cal_acc()

        self.score_list.append(round(total,6))

        # if total >= 0.65:
        #     model_path = f'{self.model_folder}/model_{self.feature_len}_{total:6.5f}_{epoch}.h5'
        #     weight_path = f'{self.model_folder}/weight_{self.feature_len}_{total:6.5f}_{epoch}.h5'
        #
        #     self.model.save_weights(weight_path)
        #     self.model.save(model_path)
        #     print(f'weight save to {model_path}')


        #threshold_map = {0:0.785, 1:0.77, 2:0.77, 3:0.77, 4:0.78}
        top_cnt =2
        top_score = self._get_top_score(self.fold)[:top_cnt]
        logger.info(f'The top#{top_cnt} score for max_bin:{get_args().max_bin}, oof:{oof_prefix}, fold#{self.fold} is:{top_score}')
        threshold = top_score[-1]
        if ( total >=threshold and epoch>=1 and total > self.max_score) or (get_args().frac<=0.1):
            #logger.info(f'Try to gen sub file for local score:{total}, and save to:{model_path}')
            self.gen_file=True
            test = self.gen_sub(self.model, f'{self.feature_len}_{total:7.6f}_{epoch}_f{self.fold}')
            len_raw_val = len(val.loc[val.bin == 0])
            self.save_stack_feature(val, test, f'./output/stacking/{oof_prefix}_{self.fold}_{total:7.6f}_{len_raw_val}_{len(val)}_{get_args().max_bin}.h5')
        else:
            logger.info(f'Only gen sub file if the local score >={threshold}, current score:{total}')

        self.max_score = max(self.max_score, total)

        logger.info(f'Epoch#{epoch},max_bin:{get_args().max_bin}, oof:{oof_prefix}, max:{self.max_score:6.5f}, acc1:{acc1:6.5f}, acc2:{acc2:6.5f}, <<<total:{total:6.5f}>>>')

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

        #frac = get_args().frac
        _, _, test = get_train_test_bert()

        label2id, id2label = get_label_id()
        input1_col = [col for col in test.columns if str(col).startswith('bert_')]
        input3_col = [col for col in test.columns if str(col).startswith('fea_')]

        logger.info(f'Input input1_col:{len(input1_col)}, input3_col:{len(input3_col)}')
        res_list = []
        for sn in tqdm(range(1+ len(test)//partition_len), desc=f'{info}:sub:total:{len(test)},partition_len:{partition_len}'):
            tmp = test.iloc[sn*partition_len: (sn+1)*partition_len]
            #print('\nbegin tmp\n', tmp.iloc[:3,:3].head())
            res = model.predict([ tmp.loc[:,input1_col], np.zeros_like(tmp.loc[:,input1_col]) ])
            res = pd.DataFrame(res, columns=label2id.keys(), index=tmp.index)
            #print('\nend tmp\n', res.iloc[:3, :3].head())
            res_list.append(res)

        res = pd.concat(res_list)
        raw_predict = res.copy()
        #print('\nafter concat\n', res.iloc[:3, :3].head())
        res['id'] = res.index
        res.index.name = 'id'
        res.to_pickle(f'./output/tmp_sub.pkl')

        res['bin'] = res.id.apply(lambda val: int(val.split('_')[1]))
        #print('\nend res\n', res.iloc[:3, :3].head())



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
        res_0 = res_0.drop(columns=['id','bin'], axis=1, errors='ignore')

        for name, res in [('single',res_0), (f'mean_{res_mean_len}', res_mean)]:
            res = res.copy()
            #logger.info(f'{name} Check:\n{res.iloc[:3,:num_classes].sum(axis=1)}')

            res['label1'] = res.iloc[:, :num_classes].idxmax(axis=1)

            # Exclude top#1
            for index, col in res.label1.items():
                res.loc[index, col] = np.nan

            res['label2'] = res.iloc[:, :num_classes].idxmax(axis=1)


            for col in ['label1','label2']:
                res[col] = res[col].replace(id2label)

            info = info.replace('.','')
            sub_file = f'./output/sub/v19_{info}_{name}.csv'
            res[['label1', 'label2']].to_csv(sub_file)
            logger.info(f'Sub file save to :{sub_file}')

        logger.info(f'res_0 Check:\n{res_0.iloc[:3, :num_classes].sum(axis=1)}')

        return raw_predict #res.drop(columns=['id','bin'], axis=1, errors='ignore')

    @staticmethod
    def _get_top_score(fold):
        from glob import glob
        file_list = sorted(glob(f'./output/stacking/{oof_prefix}_{fold}_*.h5'), reverse=True)
        score_list = [float(file.split('_')[2].replace('.h5', '')) for file in file_list]
        logger.info(f'Score list for {fold} is {score_list}')
        return score_list if score_list else [0]

if __name__ == '__main__':
    args = get_args()
    args.func(args)

"""

nohup python -u ./core/bert.py --frac=0.1  train_base  > test.log 2>&1 &


nohup python -u ./core/bert.py --fold=4 --max_bin=2 train_base  > test_4.log 2>&1 &

python -u ./core/bert.py --max_bin=2 train_base 

nohup python -u ./core/bert.py train_base  > test.log 2>&1 &

nohup python -u ./core/bert.py train_base  > extend_bert_mean_bin_1.log 2>&1 &

nohup python -u ./core/bert.py boost_train 10 >> boost_1.log 2>&1 &

"""