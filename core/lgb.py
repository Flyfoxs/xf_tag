import sys
import os
from multiprocessing import Process

from sklearn.model_selection import StratifiedKFold


import lightgbm as lgb

from core.feature import *
from core.conf import *
import keras
import os

def get_train_test_lgb():
    frac = get_args().frac
    logger.info(f'frac={frac}')

    data = get_feature_manual(10)
    raw = get_raw_data()
    logger.info(f'Shape before merge:raw:{raw.shape}, data:{data.shape}')
    data = pd.merge(data, raw, how='left', on='app_id')
    logger.info(f'Shape after merge data:{data.shape}')
    data.index = data.app_id_ex
    data['bin'] = np.nan

    from core.ensemble import get_feature_oof

    # oof = get_feature_oof(2).iloc[:, :num_classes].add_prefix('fea_oof')
    # oof['bin'] = get_feature_oof(2)['bin']
    # oof['app_id_ex_bin'] = oof.index
    # oof['app_id_ex'] = pd.Series(oof.index).apply(lambda val: val[:-2]).values
    # oof['app_id'] = pd.Series(oof.index).apply(lambda val: val.split('_')[0]).values
    #
    # logger.info(f'Shape before merge:oof:{oof.shape}, data:{data.shape}')
    #
    # data = pd.merge(data, oof, how='inner', on='app_id')
    # logger.info(f'Shape after merge oof and data:{data.shape}')

    # if 'app_des' in raw: del raw['app_des']
    # if 'app_id' in raw: del raw['app_id']
    # del raw['len_']
    logger.info(f'Shape before merge:oof:{raw.shape}, data:{data.shape}')
    #data.index = data.app_id_ex_bin
    # data['app_id_ex_bk'] = data['app_id_ex']
    # data['app_id_ex'] = data['app_id_ex'].str[:-2]



    # #Keep all the bin group, if it's test data
    # data = data.loc[(data.bin<=max_bin) | (pd.isna(data.type_id))]
    #
    # timed_bolck('Remove gan data, and len is less then 100')
    #
    # data = data.loc[ (data.bin == 0) | (data['len_'] >= 100) ]
    #
    # logger.info(f'Train max_bin:{max_bin},Total Bin distribution:\n{data.bin.value_counts().sort_index()}')

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


def train_ex(args):
    train_lgb(args.fold)

def train_lgb( fold=0, drop_list=[], args={}):

    train_data, y, X_test = get_train_test_lgb()
    num_class = num_classes

    feature_importance_df = pd.DataFrame()

    max_iteration = 0
    min_iteration = 99999

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

    with timed_bolck(f'Training#{fold}'):
        for train_idx, val_idx in [list(folds.split(train_data.values, y))[fold]]:
            import gc
            gc.collect()
            with timed_bolck(f'Fold#{fold}'):
                # val_idx = filter_index(val_idx)
                #print(train_data.shape,trn_idx.shape, val_idx.shape , X_test.shape,trn_idx.max(), val_idx.max() )
                train_x, train_y, val_x, val_y \
                    = train_data.iloc[train_idx], y.iloc[train_idx], train_data.iloc[val_idx], y.iloc[val_idx]
                feature_cnt = train_data.shape[0], train_x.shape[1]
                logger.info(f"fold n°{fold} BEGIN, all_train:{feature_cnt}, train:{train_x.shape}, val:{val_x.shape}, test:{X_test.shape}" )
                trn_data = lgb.Dataset(train_x, train_y)
                val_data = lgb.Dataset(val_x, val_y , reference=trn_data)


                params = {
                    # 'nthread': -1,
                    # 'verbose':-1,
                    # 'num_leaves': 128,
                    #### 'min_data_in_leaf': 90,
                    # 'feature_fraction':0.5,
                    # 'lambda_l1': 0.1,
                    # 'lambda_l2': 10,
                    #  'max_depth': 6,
                    #
                    # 'learning_rate': 0.1,
                    # 'bagging_fraction': 0.7,

                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'num_class': num_class,
                    #'random_state': 2019,
                    # 'device':'gpu',
                    # 'gpu_platform_id': 1, 'gpu_device_id': 0
                }
                params = dict(params, **args)

                logger.info(params)

                num_round = 30000
                #num_round = 10
                verbose_eval = 5
                with timed_bolck(f'Train#{fold}'):
                    clf = lgb.train(params,
                                    trn_data,
                                    num_round,
                                    valid_sets=[trn_data, val_data],
                                    #feval=lgb_f1_score,
                                    verbose_eval=verbose_eval,
                                    early_stopping_rounds=10)
                    logger.info(f'best_iteration:{clf.best_iteration}')

                oof = clf.predict(val_x, num_iteration=clf.best_iteration)
                oof = pd.DataFrame(oof, index=val_x.index, columns=[str(i) for i in range(num_classes)])
                oof.index.name = 'id'
                acc1, acc2, total = accuracy(oof,val_y)

                logger.info(f'score list:acc1:{acc1}, acc2:{acc2}, total:<<<{total}>>>')

                fold_importance_df = pd.DataFrame()
                fold_importance_df["feature"] = X_test.columns
                fold_importance_df["importance"] = clf.feature_importance()
                fold_importance_df["fold"] = fold + 1
                feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

                predictions = clf.predict(X_test, num_iteration=clf.best_iteration)

        predictions = pd.DataFrame(predictions, index=X_test.index, columns=[str(i) for i in range(num_classes)])
        predictions.index.name = 'id'
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)



        return oof, predictions, feature_importance_df


if __name__ == '__main__':
    args = get_args()
    args.func(args)

"""

nohup python -u ./core/lgb.py --frac=1 train_ex  > test.log 2>&1 &


"""