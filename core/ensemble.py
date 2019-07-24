import pandas as pd

from core.conf import *
from core.feature import *

@lru_cache()
def get_top_file(fold):
    from glob import glob
    file_list = sorted(glob(f'./output/stacking/{oof_prefix}_{fold}_*.h5'), reverse=True)
    return file_list

@lru_cache()
@timed()
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
    train['app_id'] = train.index.str[:32].values
    train = train.sort_values(['app_id','bin', 'label'])
    old_len = len(train)
    train = train.drop_duplicates(['app_id','bin'])
    logger.warning(f'There are {old_len - len(train)} records are removed from oof, since one sample have 2 labels')
    train = train.groupby(train.index).mean().sort_index()
    #label2id, id2label = get_label_id()
    #train.columns = [id2label[col] if col in id2label else col  for col in train.columns ]
    train['bin'] = pd.Series(train.index).apply(lambda val: val[-1]).values
    train['app_id_ex'] = pd.Series(train.index).apply(lambda val: '_'.join(val.split('_')[:-1])).values
    train['app_id'] = pd.Series(train.index).apply(lambda val: val.split('_')[0]).values

    test = pd.concat(test_list)

    test['bin'] = pd.Series(test.index).apply(lambda val: val[-1]).values
    test['app_id_ex'] = pd.Series(test.index).str[:32]
    test['app_id'] = pd.Series(test.index).str[:32]

    test = test.loc[test.bin==0].groupby('app_id').mean().sort_index()

    oof = pd.concat([train, test])

    return oof


def gen_sub_mean(top, weight=1):
    file_list = []
    for fold in range(5):
        tmp = get_top_file(fold)
        # print(tmp)
        file_list = file_list + tmp[:top]
    #print(file_list)
    df_list = []
    for file in tqdm(file_list):

        cur_weight = weight if weight >=0 else get_best_weight(file)

        tmp = pd.read_hdf(file, 'test')
        tmp['bin'] = pd.Series(tmp.index).str[-1].astype(int).values
        tmp['id'] = pd.Series(tmp.index).str[:32].values

        col_list = tmp.columns[:num_classes]
        #logger.info(f'Adjust the sub with weight:{cur_weight:3.2f} for file:{file}')
        tmp.loc[tmp.bin == 0, col_list] = tmp.loc[tmp.bin == 0, col_list] * cur_weight
        tmp.loc[tmp.bin == 1, col_list] = tmp.loc[tmp.bin == 1, col_list] * (1 - cur_weight)
        logger.info(f'{file}:{tmp.shape}\n{tmp.bin.value_counts()}')
        tmp = tmp.loc[tmp.bin.isin([0,1])].groupby('id').mean() #
        df_list.append(tmp)

    res = pd.concat(df_list)
    total = res.copy()

    res = res.groupby('id').mean().sort_index()

    #logger.info(f'The shape of ensemble sub is:{res.shape}')

    res['label1'] = res.iloc[:, :num_classes].idxmax(axis=1)

    # Exclude top#1
    for index, col in res.label1.items():
        res.loc[index, col] = np.nan

    res['label2'] = res.iloc[:, :num_classes].idxmax(axis=1)

    label2id, id2label = get_label_id()
    for col in ['label1', 'label2']:
        res[col] = res[col].replace(id2label)

    # info = info.replace('.','')
    sub_file = f'./output/sub/mean_{len(file_list)}_{int(weight*100):03}.csv'
    res[['label1', 'label2']].to_csv(sub_file)
    logger.info(f'Sub file save to :{sub_file}')

    return total



@timed()
def get_best_weight(file):
    df = pd.read_hdf(file, 'train')
    df['bin'] = df.index.str[-1].astype(int)

    col_list = df.columns[:num_classes]
    #print(col_list)
    df['bin'] = df.index.str[-1].astype(int)
    df['app_id'] = df.index.str[:32]

    if len(df.loc[df.bin==1]) ==0 :
        return 1

    print(df.bin.value_counts())
    df = df.sort_values(['app_id', 'bin', 'label'])
    df = df.drop_duplicates(['app_id', 'bin'])

    score ={}

    for weight in np.arange(0.7, 1.01, 0.05):
        weight = round(weight, 2)
        tmp = df.copy()
        tmp.loc[tmp.bin == 0, col_list] = tmp.loc[tmp.bin == 0, col_list] * weight
        tmp.loc[tmp.bin == 1, col_list] = tmp.loc[tmp.bin == 1, col_list] * (1 - weight)

        # tmp = tmp.loc[tmp.bin==0]
        tmp = tmp.loc[tmp.bin.isin([0, 1])]
        #print(tmp.bin.value_counts())
        tmp = tmp.groupby('app_id').mean()

        print(tmp.shape)
        tmp.label = tmp.label.astype(int)
        # print(tmp.shape)
        acc1, acc2, total = accuracy(tmp)
        score[weight] = total

    logger.info(f'Score list for file:{file}\n{score}')

    base_score = list(score.values())[-1]

    score = sorted(score.items(), key=lambda kv: kv[1])
    best_score = score[-1][-1]
    best_weight = score[-1][0]
    grow = best_score-base_score

    logger.info(f'====best_weight:{best_weight:3.2}, best_score:{best_score:6.5f}/{grow:6.5f}')
    return best_weight



if __name__== '__main__':
    from core.ensemble import *

    sub = gen_sub_mean(2, weight=-1)
    sub = gen_sub_mean(2, weight=1)
