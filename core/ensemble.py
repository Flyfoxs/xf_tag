import pandas as pd

from core.conf import *
from core.feature import *


def get_top_file(fold):
    from glob import glob
    file_list = sorted(glob(f'./output/stacking/v?_{fold}_*.h5'), reverse=True)
    return file_list


def get_feature_oof(top):
    file_list = []
    for fold in range(5):
        tmp = get_top_file(fold)
        # print(tmp)
        file_list = file_list + tmp[:top]
    print(file_list)
    train_list = []
    test_list = []
    for file in file_list:
        tmp = pd.read_hdf(file, 'train')
        train_list.append(tmp)

        tmp = pd.read_hdf(file, 'test')
        test_list.append(tmp)

    train = pd.concat(train_list)
    train = train.groupby(train.index).mean().sort_index()
    train['bin'] = pd.Series(train.index).apply(lambda val: val[-1]).values

    test = pd.concat(test_list)
    test = test.groupby(test.index).mean().sort_index()

    oof = pd.concat([train, test])

    return oof


def gen_sub_mean(top):
    file_list = []
    for fold in range(5):
        tmp = get_top_file(fold)
        # print(tmp)
        file_list = file_list + tmp[:top]
    print(file_list)
    df_list = []
    for file in file_list:
        tmp = pd.read_hdf(file, 'test')
        df_list.append(tmp)
        print(tmp.shape)
    res = pd.concat(df_list)
    total = res.copy()
    res['id'] = res.index
    res = res.groupby('id').mean().sort_index()

    res['label1'] = res.iloc[:, :num_classes].idxmax(axis=1)

    # Exclude top#1
    for index, col in res.label1.items():
        res.loc[index, col] = np.nan

    res['label2'] = res.iloc[:, :num_classes].idxmax(axis=1)

    label2id, id2label = get_label_id()
    for col in ['label1', 'label2']:
        res[col] = res[col].replace(id2label)

    # info = info.replace('.','')
    sub_file = f'./output/sub/mean_{len(file_list)}.csv'
    res[['label1', 'label2']].to_csv(sub_file)
    logger.info(f'Sub file save to :{sub_file}')

    return total