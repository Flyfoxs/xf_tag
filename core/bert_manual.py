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
n_topic = 10

#Batch size, MAX_len+ex_length, Manual, Manual GP feature cnt, frac
@lru_cache()
@timed()
def get_train_test_bert():

    frac = get_args().frac
    max_bin = get_args().max_bin
    min_len = int(SEQ_LEN*get_args().min_len_ratio)

    bert = get_feature_bert(SEQ_LEN)
    manual = get_feature_bert_wv().add_prefix('fea_')
    manual['app_id'] = manual.index.str[:32].values
    manual = manual.drop_duplicates('app_id')

    old_shape = bert.shape
    data = pd.merge(bert, manual, how='left', on=['app_id'])
    data.index = data.app_id_ex_bin

    logger.info(f'old shape:{old_shape}, new_shape:{data.shape}')

    #Keep all the bin group, if it's test data
    data = data.loc[(data.bin<=max_bin) | (pd.isna(data.type_id))]

    timed_bolck('Remove gan data, and len is less then 100')

    data = data.loc[ (data.bin == 0) | (data['len_'] >= min_len) ]

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



@timed()
def manual_train():
    #frac = args.frac
    args = get_args()
    fold = args.fold
    EPOCHS = args.epochs

    BATCH_SIZE = 128
    LR = 1e-4

    with timed_bolck(f'Prepare train data#{BATCH_SIZE}'):
        X, y, _ = get_train_test_bert()




        ##Begin to define model
        from keras_bert import load_trained_model_from_checkpoint

        model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, seq_len=SEQ_LEN, )


        from tensorflow.python import keras
        from keras_bert import AdamWarmup, calc_train_steps
        inputs = model.inputs[:2]
        dense_bert = model.get_layer('NSP-Dense').output


        decay_steps, warmup_steps = calc_train_steps(
            y.shape[0],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )

        # New input from manual

        data = get_feature_bert_wv().add_prefix('fea_')
        manual_fea_len = len([col for col in data.columns if col.startswith('fea_')])

        logger.info(f'manual_fea_len:{manual_fea_len}')
        manual_feature = keras.Input(shape=(manual_fea_len,), name='manual_feature', dtype='float32')
        inputs = inputs + [manual_feature]


        manual_feature = keras.layers.Dense(round(num_classes*0.6),  name='manual_dense', activation='relu')(manual_feature)
        manual_feature = keras.layers.Dropout(0.5)(manual_feature)
        #manual_feature = keras.layers.Dense(round(num_classes), activation='relu')(manual_feature)

        fc_ex = keras.layers.concatenate([dense_bert, manual_feature], axis=1)
        # End input from manual

        #fc_ex = keras.layers.Dense(units=1024, activation='softmax')(fc_ex)

        outputs = keras.layers.Dense(units=num_classes, activation='softmax')(fc_ex)

        model = keras.models.Model(inputs, outputs)
        model.compile(
            AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        model.summary(line_length=120)
        ##End to define model

        input1_col = [col for col in X.columns if str(col).startswith('bert_')]
        input3_col = [col for col in X.columns if str(col).startswith('fea_')]
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
        #for sn in range(5):
        input1 = train_x.loc[:, input1_col]#.astype(np.float32)
        input2 = np.zeros_like(input1)#.astype(np.int8)
        input3 = train_x.loc[:, input3_col]
        logger.info(f'NN Input1:{input1.shape}, Input2:{input2.shape}, Input3:{input3.shape}')

        logger.info(f'NN train_x:{train_x[:3]}')

        from keras_bert import get_custom_objects
        import tensorflow as tf

        with tf.keras.utils.custom_object_scope(get_custom_objects()):
            his = model.fit([input1, input2, input3], train_y,
                            validation_data = ([
                                                val_x.loc[:, input1_col],
                                                np.zeros_like(val_x.loc[:, input1_col]),
                                                val_x.loc[:, input3_col]
                                               ],
                                               val_y),
                            epochs=EPOCHS,  shuffle=True, batch_size=64,
                            callbacks=[Cal_acc(val_x, y.iloc[test_idx] )]
                      #steps_per_epoch=1000, validation_steps=10
                      )



            #gen_sub(model, X_test, sn)

    return his

class Cal_acc(Callback):

    def __init__(self, val_x, y ):
        super(Cal_acc, self).__init__()
        self.val_x , self.y = val_x, y
        self.min_len = int(SEQ_LEN*get_args().min_len_ratio)
        self.max_bin = get_args().max_bin
        self.fold = get_args().fold
        self.threshold = 0
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
        input3_col = [col for col in self.val_x.columns if str(col).startswith('fea_')]
        #model = self.model
        val = self.model.predict([self.val_x.loc[:,input1_col],
                                  np.zeros_like(self.val_x.loc[:,input1_col]),
                                  self.val_x.loc[:, input3_col]
                                  ])

        label2id, id2label = get_label_id()
        val = pd.DataFrame(val, columns=label2id.keys(), index=self.val_x.index)
        val['label'] = self.y.astype(int).replace(id2label).astype(int)
        val['bin'] = pd.Series(val.index).str[-1].values.astype(int)
        #logger.info(f'Head val#label:\n{val.label.head()}')
        res_val = val.copy()
        # res_val.to_pickle(f'./output/tmp_res_val.pkl')
        # logger.info(f'Debug file: save to ./output/tmp_res_val.pkl')

        logger.info(f'val bin:\n {val.bin.value_counts()}')
        #y2['bin'] = pd.Series(y2.index).str[-1].values.astype(int)
        for bin_list in [[0,1], [0]]:
            tmp_val = val.loc[val.bin.isin(bin_list)].copy()
            if len(tmp_val)>0:

                df_len = len(tmp_val)

                tmp_val['bin'] = pd.Series(tmp_val.index).str[-1].astype(int)
                tmp_val['app_id'] = tmp_val.index.str[:32].values
                tmp_val = tmp_val.sort_values(['app_id', 'bin', 'label'])
                #One sample can have 2 label in the original data
                tmp_val = tmp_val.drop_duplicates(['app_id', 'bin'])

                tmp_val_mean = tmp_val.groupby('app_id').mean()
                tmp_val_max = tmp_val.groupby('app_id').max()
                for name, df in [('mean', tmp_val_mean), ('max', tmp_val_max)]:
                    score_list = accuracy(df,4)
                    logger.info(f'Val({name})#{len(df)}/{df_len}, bin#{bin_list}, score_list:{score_list}')
            else:
                logger.info(f'Can not find bin:{bin_list} in val')
        return score_list, res_val


    def on_train_end(self, logs=None):
        grow= max(self.score_list) - self.threshold
        logger.info(f'Fold:{self.fold}, max:{max(self.score_list):7.6f}/{grow:+6.5f}, at {np.argmax(self.score_list)}/{len(self.score_list)-1}, Train his:{self.score_list}, max_bin:{self.max_bin}, min_len:{self.min_len}, gen_file:{self.gen_file}')
        logger.info(f'Input args:{get_args()}')

    def on_epoch_end(self, epoch, logs=None):
        print('\n')
        score_list, val = self.cal_acc()
        total = score_list[1]
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
        self.threshold = top_score[-1] if len(top_score) == top_cnt else 0
        logger.info(f'The top#{top_cnt} score for max_bin:{get_args().max_bin}, epoch:{epoch}, oof:{oof_prefix}, fold#{self.fold} is:{top_score}, cur_score:{total}, threshold:{self.threshold}')
        if ( round(total,4) > round(self.threshold,4) and epoch>=1 and total > self.max_score) or (get_args().frac<=0.1):
            #logger.info(f'Try to gen sub file for local score:{total}, and save to:{model_path}')
            self.gen_file=True
            grow = max(self.score_list) - self.threshold
            logger.info(f'Fold:{self.fold}, epoch:{epoch}, MAX:{max(self.score_list):7.6f}/{grow:+6.5f}, threshold:{self.threshold}, score_list:{self.score_list}' )
            test = self.gen_sub(self.model, f'{self.feature_len}_{total:7.6f}_{epoch}_f{self.fold}')
            len_raw_val = len(val.loc[val.bin == 0])
            min_len_ratio = get_args().min_len_ratio
            oof_file = f'./output/stacking/{oof_prefix}_{self.fold}_{total:7.6f}_{len_raw_val}_{len(val):05}_b{get_args().max_bin}_e{epoch}_m{min_len_ratio:2.1f}_L{SEQ_LEN:03}_w{self.window}.h5'
            self.save_stack_feature(val, test, oof_file)
        else:
            logger.info(f'Epoch:{epoch}, only gen sub file if the local score >{self.threshold}, current score:{total}')

        self.max_score = max(self.max_score, total)

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
            res = model.predict([ tmp.loc[:,input1_col], np.zeros_like(tmp.loc[:,input1_col]), tmp.loc[:,input3_col] ])
            res = pd.DataFrame(res, columns=label2id.keys(), index=tmp.index)
            #print('\nend tmp\n', res.iloc[:3, :3].head())
            res_list.append(res)

        res = pd.concat(res_list)
        res['bin'] = res.index.str[-1].values.astype(int)
        raw_predict = res.copy()
        return raw_predict #res.drop(columns=['id','bin'], axis=1, errors='ignore')

    @staticmethod
    def _get_top_score(fold):
        from glob import glob
        file_list = sorted(glob(f'./output/stacking/{oof_prefix}_{fold}_*.h5'), reverse=True)
        score_list = [float(file.split('_')[2].replace('.h5', '')) for file in file_list]
        logger.info(f'Score list for {fold} is {score_list}')
        return score_list if score_list else [0]

if __name__ == '__main__':
    FUNCTION_MAP = {'manual_train': manual_train,
                    }

    args = get_args()

    func = FUNCTION_MAP[args.command]
    func()

"""

nohup python -u ./core/bert_manual.py manual_train  > test.log 2>&1 &


nohup python -u ./core/bert.py --fold=4 --max_bin=2 train_base  > test_4.log 2>&1 &

python -u ./core/bert.py --max_bin=2 train_base 

nohup python -u ./core/bert.py train_base  > test.log 2>&1 &

nohup python -u ./core/bert.py train_base  > extend_bert_mean_bin_1.log 2>&1 &

nohup python -u ./core/bert.py boost_train 10 >> boost_1.log 2>&1 &

"""