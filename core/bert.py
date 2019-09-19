
from multiprocessing import Process





from core.feature import *
from core.conf import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_KERAS'] = '1'

oof_prefix = get_args().version
SEQ_LEN = get_args().seq_len  #randrange(128, 180) #-randrange(0, 5)*8
BATCH_SIZE = get_args().batch_size

#Batch size, MAX_len+ex_length, Manual, Manual GP feature cnt, frac
@lru_cache()
@timed()
def get_train_test_bert():

    frac = get_args().frac
    max_bin = get_args().max_bin
    min_len = int(SEQ_LEN*get_args().min_len_ratio)

    data = get_feature_bert(SEQ_LEN)

    #Keep all the bin group, if it's test data
    data = data.loc[(data.bin<=max_bin) | (pd.isna(data.type_id))]

    with timed_bolck(f'Remove gan data, and len is less then {min_len}'):
        data = data.loc[ (data.bin == 0) | (data['len_'] >= min_len) ]
        logger.info(f'Train max_bin:{max_bin},Total Bin distribution:\n{data.bin.value_counts().sort_index()}')

    data = data.sort_index()
    logger.info(f'Head of the data:\n, {data.iloc[:3,:3]}')

    train_data = data.loc[pd.notna(data.type_id)].sample(frac=frac, random_state=2019)
    labels = train_data.type_id.values.tolist()
    logger.info(f'Train Bin distribution:\n{train_data.bin.value_counts().sort_index()}')

    test_data =  data.loc[pd.isna(data.type_id)].sample(frac=1, random_state=2019)

    trial = get_args().trial
    logger.info(f'Test Bin distribution#{trial}:\n{test_data.bin.value_counts().sort_index()}')

    if trial > 0:
        test_data = test_data.loc[test_data.index.str[-1]=='0']


    logger.info(f'Train:{train_data.shape} Test#{trial}:{test_data.shape}, frac:{frac}')

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
def filter_short_desc(X, y):
    X = X.copy().reset_index()
    bert_cols = [col for col in X.columns if str(col).startswith('bert_')]
    bert = X.loc[:, bert_cols]
    bert_len = bert.where(bert > 0).count(axis=1)
    old_len = len(bert_len)
    min_len = int(SEQ_LEN*get_args().min_len_ratio)
    bert_len = bert_len.loc[bert_len >= min_len]
    logger.info(f'Filter {old_len - len(bert_len)} records from {old_len} by threshold {min_len}')

    return X.iloc[bert_len.index], y[bert_len.index]


@timed()
def train_base():
    args = get_args()
    #frac = args.frac
    fold = args.fold
    EPOCHS = args.epochs


    LR = 2e-5

    with timed_bolck(f'Prepare train data#{BATCH_SIZE}, LR:{LR}'):
        X, y, _ = get_train_test_bert()

        ##Begin to define model
        from keras_bert import load_trained_model_from_checkpoint

        logger.info(f'Start to train base on checkpoint:{config_path}')
        bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path,  seq_len=SEQ_LEN, )

        for l in bert_model.layers:
            l.trainable = True
        from tensorflow.python import keras
        #from keras_bert import  calc_train_steps

        x1_in = keras.layers.Input(shape=(None,))
        x2_in = keras.layers.Input(shape=(None,))

        x = bert_model([x1_in, x2_in])

        x = keras.layers.Lambda(lambda x: x[:, 0])(x)

        p = keras.layers.Dense(num_classes, activation='sigmoid')(x)

        #from keras import Model
        model = keras.models.Model([x1_in, x2_in], p)


        model.compile(
            optimizer=keras.optimizers.Adam(lr=LR), # AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        model.summary()
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
        train_idx, test_idx = split_df_by_index(X,fold)

        logger.info(f'Shape train_x.loc[:, input1_col].iloc[:,0]: {X.loc[:, input1_col].iloc[:,0].shape}')
        train_x, train_y, val_x, val_y = \
            X.iloc[train_idx], Y_cat[train_idx], X.iloc[test_idx], Y_cat[test_idx]

        logger.info(f'get_train_test output: train_x:{train_x.shape}, train_y:{train_y.shape}, val_x:{val_x.shape} ')

        #train_x, train_y = filter_short_desc(train_x, train_y)

        input1 = train_x.loc[:, input1_col]#.astype(np.float32)
        input2 = np.zeros_like(input1)#.astype(np.int8)

        logger.info(f'NN train_x:{train_x[:3]}')
        min_len_ratio = get_args().min_len_ratio
        max_bin = get_args().max_bin
        logger.info(f'NN Input1:{input1.shape}, Input2:{input2.shape}, SEQ_LEN:{SEQ_LEN}, min_len_ratio:{min_len_ratio}, bin:{max_bin} ')

        from keras_bert import get_custom_objects
        import tensorflow as tf
        with tf.keras.utils.custom_object_scope(get_custom_objects()):
            his = model.fit([input1, input2], train_y,
                            validation_data = ([val_x.loc[:, input1_col], np.zeros_like(val_x.loc[:, input1_col])], val_y),
                            epochs=EPOCHS,  shuffle=True, batch_size=BATCH_SIZE,
                            callbacks=[Cal_acc( val_x, y.iloc[test_idx] )]
                      #steps_per_epoch=1000, validation_steps=10
                      )



            #gen_sub(model, X_test, sn)

    return his

from tensorflow.python.keras.callbacks import Callback
class Cal_acc(Callback):

    def __init__(self, val_x, y):
        super(Cal_acc, self).__init__()
        self.val_x , self.y = val_x, y
        self.min_len = int(SEQ_LEN*get_args().min_len_ratio)
        self.max_bin = get_args().max_bin
        self.fold = get_args().fold
        self.threshold = 0
        self.feature_len = self.val_x.shape[1]
        self.cur_epoch = 0
        self.version = get_args().version
        self.trial = get_args().trial

        self.max_score = 0

        self.score_list = np.zeros(get_args().epochs)
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
        tmp_val = self.val_x.loc[:,input1_col]
        tmp_y = self.y
        val = self.model.predict([tmp_val, np.zeros_like(tmp_val)])

        label2id, id2label = get_label_id()
        val = pd.DataFrame(val, columns=label2id.keys(), index=tmp_val.index)
        val['label'] = tmp_y.astype(int).replace(id2label).astype(int)
        val['bin'] = pd.Series(val.index).str[-1].values.astype(int)
        #logger.info(f'Head val#label:\n{val.label.head()}')
        res_val = val.copy()
        # res_val.to_pickle(f'./output/tmp_res_val.pkl')
        # logger.info(f'Debug file: save to ./output/tmp_res_val.pkl')

        num_labels = 10
        df_score = val.loc[val.bin==0]
        score_list = accuracy(df_score, num_labels, f'no{self.cur_epoch},b{self.max_bin},{self.version}')

        logger.info(f'{len(df_score)}/{len(res_val)}, fold:{self.fold}, score for label1-f{num_labels}:{score_list}')

        return score_list,res_val

    @timed()
    def cal_acc_ex(self):
        input1_col = [col for col in self.val_x.columns if str(col).startswith('bert_')]

        if self.trial==0:
            check_type_list =['val']
        for type_ in tqdm(check_type_list,desc='cal_acc_ex'):
            tmp_val ,tmp_y = self.get_tmp_val_test(type_)
            tmp_val = tmp_val.loc[:, input1_col]

            val = self.model.predict([tmp_val, np.zeros_like(tmp_val)])

            label2id, id2label = get_label_id()
            val = pd.DataFrame(val, columns=label2id.keys(), index=tmp_val.index)
            val['label'] = tmp_y.astype(int).replace(id2label).astype(int)
            val['bin'] = pd.Series(val.index).str[-1].values.astype(int)
            # logger.info(f'Head val#label:\n{val.label.head()}')
            res_val = val.copy()
            # res_val.to_pickle(f'./output/tmp_res_val.pkl')
            # logger.info(f'Debug file: save to ./output/tmp_res_val.pkl')

            num_labels = 10
            df_score = val.loc[val.bin == 0]
            score_list = accuracy(df_score, num_labels, f'ex{self.cur_epoch},{self.version},b{self.max_bin},{type_}')

            logger.info(f'===cal_acc_ex{self.cur_epoch}:{type_}==={len(df_score)}/{len(res_val)}, fold:{self.fold}, score for label1-f{num_labels}:{score_list}')

        return score_list, res_val


    @lru_cache()
    @timed()
    def get_tmp_val_test(self, type_):
        _, _, test_all = get_train_test_bert()

        test = test_all.loc[pd.Series(test_all.index).str.startswith(type_).values]

        test = test.loc[(pd.Series(test.index).str[-1]=='0').values]

        logger.info(f'Split {type_}, {len(test)} rows from {len(test_all)}')

        test=test.copy()
        type_ = 'x'*6 + pd.Series(test.index).str[:6]
        test.index = 'x'*6 + pd.Series(test.index).str[6:]

        from spider.mi import get_train_ph2_index
        train_ph2 =  get_train_ph2_index()
        #final = final.loc[final.type_id.str.len() >= 1]
        train_ph2.index = 'x'*6 + train_ph2['id'].str[6:]
        #Align label with input test
        index_old = test.index.copy()
        test.index = pd.Series(test.index).apply(lambda val: val[:32])

        label = train_ph2.type_id.loc[test.index.values].str[:6] #type_id len is 6

        #Rollback index change
        test.index = index_old
        label.index = index_old

        test = test.loc[pd.notna(label).values]
        label = label.dropna()
        print('test, label, type_', test.shape, label.shape, type_.shape)
        return test, label#, type_


    def on_train_end(self, logs=None):
        grow= max(self.score_list) - self.threshold
        cut_ratio = get_args().cut_ratio
        logger.info(f'Train END: Fold:{self.fold}, max:{max(self.score_list):7.6f}/{grow:+6.5f}, at {np.argmax(self.score_list)}/{len(self.score_list)-1}, his:{self.score_list}, max_bin:{self.max_bin}, cut:{cut_ratio}, min_len:{self.min_len:03}, SEQ_LEN:{SEQ_LEN:03}, threshold:{self.threshold:7.6f}, gen_file:{self.gen_file}')
        logger.info(f'Input args:{get_args()}')

    def on_epoch_end(self, epoch, logs=None):
        self.cur_epoch = epoch
        print('\n')
        _, _ = self.cal_acc_ex()

        if self.trial > 0:
            return 0
        else:
            score_list, val = self.cal_acc()
            total = score_list[1]

            self.score_list[epoch] = round(total, 6)
            #threshold_map = {0:0.785, 1:0.77, 2:0.77, 3:0.77, 4:0.78}
            top_cnt =2
            top_score = self._get_top_score(self.fold)[:top_cnt]
            self.threshold = top_score[0] if len(top_score) >  0 else 0
            logger.info(f'The top#{top_cnt} score for max_bin:{get_args().max_bin}, epoch:{epoch}, oof:{oof_prefix}, fold#{self.fold} is:{top_score}, cur_score:{total}, threshold:{self.threshold}')
            if ( round(total,4) > round(self.threshold,4)
                 and (epoch>=3 or self.threshold > 0 )
                 and total > self.max_score
                ) :
                #logger.info(f'Try to gen sub file for local score:{total}, and save to:{model_path}')
                self.gen_file=True
                grow = max(self.score_list) - self.threshold
                logger.info(f'Fold:{self.fold}, epoch:{epoch}, MAX:{max(self.score_list):7.6f}/{grow:+6.5f}, threshold:{self.threshold}, score_list:{self.score_list}' )
                test = self.gen_sub(self.model, f'{self.feature_len}_{total:7.6f}_{epoch}_f{self.fold}')
                len_raw_val = len(val.loc[val.bin == 0])
                min_len_ratio = get_args().min_len_ratio
                oof_file = f'./output/stacking/{oof_prefix}_{self.fold}_{total:7.6f}_{len_raw_val}_{len(val):05}_b{get_args().max_bin}_e{epoch}_{self.batch_id}_m{min_len_ratio:2.1f}_L{SEQ_LEN:03}.h5'
                self.save_stack_feature(val, test, oof_file)
            else:
                logger.info(f'Epoch:{epoch}, only gen sub file if the local score >{self.threshold}, current score:{total}, threshold:{self.threshold}, max_score:{self.max_score}')

            self.max_score = max(self.max_score, total, 0.82)

            logger.info(f'Epoch#{epoch} END,max_bin:{get_args().max_bin}, oof:{oof_prefix}, max:{self.max_score:6.5f}, score:{score_list}, Fold:{self.fold},')

            print('\n')

            return round(total, 5)

    @staticmethod
    @timed()
    def save_stack_feature(train: pd.DataFrame, test: pd.DataFrame, file_path):
        train.bin = train.bin.astype(int)
        test.bin = test.bin.astype(int)
        train.to_hdf(file_path, 'train', mode='a')
        test.to_hdf(file_path, 'test', mode='a')
        logger.info(f'OOF file save to :{file_path}')
        return train, test


    @timed()
    #./output/model/1562899782/model_6114_0.65403_2.h5
    def gen_sub(self, model , info='bert_' , partition_len = 5000):

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
        res['bin'] = res.index.str[-1].values.astype(int)
        raw_predict = res.copy()

        with timed_bolck(f'Try to gen sub file for fold#{self.fold}'):
            #print('\nafter concat\n', res.iloc[:3, :3].head())
            res['id'] = res.index
            res.index.name = 'id'
            # res.to_pickle(f'./output/tmp_sub.pkl')


            #print('\nend res\n', res.iloc[:3, :3].head())



            res_mean = res.copy(deep=True)
            res_mean['id'] = res_mean.id.apply(lambda val: val.split('_')[0])
            res_mean.index.name = 'index'
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

                # info = info.replace('.','')
                # sub_file = f'./output/sub/v19_{info}_{name}.csv'
                # res[['label1', 'label2']].to_csv(sub_file)
                # logger.info(f'Sub file save to :{sub_file}')

            #logger.info(f'res_0 Check:\n{res_0.iloc[:3, :num_classes].sum(axis=1)}')

        return raw_predict #res.drop(columns=['id','bin'], axis=1, errors='ignore')

    @staticmethod
    def _get_top_score(fold):
        from glob import glob
        file_list = sorted(glob(f'./output/stacking/{oof_prefix}_{fold}_*.h5'), reverse=True)
        score_list = [float(file.split('_')[2].replace('.h5', '')) for file in file_list]
        logger.info(f'Score list for fold#{fold} is {score_list}')
        return score_list if score_list else [0]

if __name__ == '__main__':
    FUNCTION_MAP = {'train_base': train_base,
                    }

    args = get_args()

    func = FUNCTION_MAP[args.command]
    func()

"""

nohup python -u ./core/bert.py --frac=0.1  train_base  > test.log 2>&1 &

nohup python -u ./core/bert.py --fold=4 --max_bin=2 train_base  > test_4.log 2>&1 &

python -u ./core/bert.py --max_bin=2 train_base 

nohup python -u ./core/bert.py train_base  > test.log 2>&1 &

nohup python -u ./core/bert.py train_base  > extend_bert_mean_bin_1.log 2>&1 &

nohup python -u ./core/bert.py boost_train 10 >> boost_1.log 2>&1 &

"""