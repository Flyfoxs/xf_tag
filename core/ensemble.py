import pandas as pd

from core.conf import *
from core.feature import *

static_list = [
# './output/stacking/v6_0_0.804024_6006_10605_b1_e1_m50.h5',
# './output/stacking/v6_0_0.804532_6006_10605_b2_e1_m50.h5',
# './output/stacking/v6_1_0.789411_5918_10451_b2_e1_m50.h5',
# './output/stacking/v6_1_0.792143_5918_10451_b1_e1_m50.h5',
# './output/stacking/v6_2_0.790876_6237_11068_b1_e1_m50.h5',
# './output/stacking/v6_2_0.791542_6237_11068_b1_e1_m50.h5',
# './output/stacking/v6_3_0.799421_5996_10562_b1_e1_m50.h5',
# './output/stacking/v6_3_0.801635_5996_10562_b1_e1_m50.h5',
# './output/stacking/v6_4_0.765271_6977_12388_b4_e1_m20.h5',
# './output/stacking/v6_4_0.766215_6977_06977_b0_e1_m20.h5',
]
@lru_cache()
def get_top_file(fold):
    from glob import glob
    file_list = sorted(glob(f'./output/stacking/{oof_prefix}_{fold}_*.h5'), reverse=True)

    if static_list:
        file_list = [ file for file in file_list if file in static_list]
    return file_list

@lru_cache()
@timed()
def get_feature_oof(top=2, weight=1):
    file_list = []
    for fold in range(5):
        tmp = get_top_file(fold)
        if len(tmp) < top:
            raise Exception(f'At least need {top} files for fold:{fold}')
        file_list = file_list + tmp[:top]
    print(file_list)
    train_list = []
    test_list = []

    for file in file_list:
        cur_weight = weight if weight > 0 else get_best_weight(file)

        #Train begin
        tmp = pd.read_hdf(file, 'train')
        col_list = tmp.columns[:num_classes]
        tmp['app_id'] = tmp.index.str[:32].values
        tmp['bin'] = tmp.index.str[-1].values.astype(int)
        tmp = tmp.sort_values(['app_id', 'bin', 'label'])
        tmp = tmp.drop_duplicates(['app_id', 'bin'])

        tmp.loc[tmp.bin == 0, col_list] = tmp.loc[tmp.bin == 0, col_list] * cur_weight
        tmp.loc[tmp.bin == 1, col_list] = tmp.loc[tmp.bin == 1, col_list] * (1 - cur_weight)
        tmp.label = tmp.label.astype(int)
        tmp = tmp.loc[tmp.bin.isin([0, 1])].groupby('app_id').mean()

        train_list.append(tmp)

        #Test begin
        tmp = pd.read_hdf(file, 'test')
        tmp['app_id'] = tmp.index.str[:32].values
        tmp['bin'] = tmp.index.str[-1].values.astype(int)
        tmp.loc[tmp.bin == 0, col_list] = tmp.loc[tmp.bin == 0, col_list] * cur_weight
        tmp.loc[tmp.bin == 1, col_list] = tmp.loc[tmp.bin == 1, col_list] * (1 - cur_weight)
        tmp = tmp.loc[tmp.bin.isin([0, 1])].groupby('app_id').mean()
        test_list.append(tmp)

    train = pd.concat(train_list)


    test = pd.concat(test_list)

    oof = pd.concat([train, test])

    oof = oof.groupby(oof.index).mean()
    del oof['bin']
    oof.label = oof.label.fillna(0).astype(int).astype(str)
    return oof

@timed()
def gen_sub_file(res, file_name, topn=2):
    res = res.copy()
    res_raw = res.copy()

    for i in tqdm(range(1, 1+topn), desc=f'Cal label#1-{topn} value for res:{res.shape}'):
        res.loc[:, f'label{i}'] = res.iloc[:, :num_classes].idxmax(axis=1)
        res_raw.loc[:, f'label{i}'] = res.loc[:, f'label{i}']

        for index, col in res[f'label{i}'].items():
            res.loc[index, col] = np.nan


    if file_name:
        res.index.name = 'id'
        sub_file = f'./output/sub/{oof_prefix}_{file_name}'
        res[['label1', 'label2']].to_csv(sub_file)
        logger.info(f'Sub file save to :{sub_file}')

    return res_raw



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

    for weight in tqdm(np.arange(0.7, 1.01, 0.05), desc=f'Cal best for {file}'):
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
        score_list = accuracy(tmp)
        logger.info(f'weight:{weight}, score_list:{score_list}. File:{file}')
        total = score_list[1]
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
    for top in [2, 3]:
        for weight in [ 0,  0.95, 1]:
            with timed_bolck(f'Cal sub for top:{top}, weight:{weight:3.2f}'):
                res = get_feature_oof(top, weight)
                train = res.loc[res.label != '0']
                score_list = accuracy(train)
                total = score_list[1]
                file_name = f'mean_top{top}_{int(weight * 100):03}_{int(total*10**6):06}.csv'
                res = gen_sub_file(res.loc[res.label == '0'], file_name)
                #logger.info(f'Sub file save to:{file_name}')





"""
nohup python -u ./core/ensemble.py > ensemble.log 2>&1 &
"""



