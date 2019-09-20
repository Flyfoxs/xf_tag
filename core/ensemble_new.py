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
def get_top_file(fold,version):
    from glob import glob
    file_list = sorted(glob(f'./output/stacking/{version}_{fold}_*.h5'), reverse=True)

    if static_list:
        file_list = [ file for file in file_list if file in static_list]
    return file_list

@lru_cache()
def get_file_list(version, top=2,):
    file_list  = []
    for fold in range(5):
        tmp = get_top_file(fold, version)
        if len(tmp) < top:
            logger.warning(f'At least need {top} files for fold:{fold}')
        file_list = file_list + tmp[:top]
    return tuple(file_list)

@lru_cache()
@timed()
def get_feature_oof(file_list, weight=1,base_train=True):

    train_list = []
    test_list = []

    for file in tqdm(file_list,f'gen oof from {len(file_list)} files'):
        cur_weight = weight if weight > 0 else get_best_weight(file, base_train=base_train)

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
    print('oof, before=', oof.shape)
    oof = oof.groupby(oof.index).mean()
    print('oof, after=', oof.shape)
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
        from spider.mi import get_train_ph2_index
        train_ph2 = get_train_ph2_index()

        res_bk = res.copy().loc[~res.index.str[6:].isin(train_ph2.id.str[6:].values)]
        for res in [res, res_bk]:
            res.index.name = 'id'
            sub_file = f'./output/sub/{len(res)}_{file_name}'
            res[['label1', 'label2']].to_csv(sub_file)
            logger.info(f'Sub file save to :{sub_file}')

    return res_raw



@timed()
def get_best_weight(file, base_train):
    import pandas as pd
    if base_train:
        df = pd.read_hdf(file, 'train')
    else:
        df = pd.read_hdf(file, 'test')

        from spider.mi import get_train_ph2_index
        ph2_train = get_train_ph2_index()
        ph2_train = ph2_train.set_index('id')
        df = df.loc[pd.Series(df.index).str[:32].isin(ph2_train.index).values]
        df['label'] = ph2_train.loc[pd.Series(df.index).str[:32]].type_id.str[:6].values.astype(int)


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
        # print(tmp.label.head(3))
        tmp.loc[tmp.bin == 0, col_list] = tmp.loc[tmp.bin == 0, col_list] * weight
        tmp.loc[tmp.bin == 1, col_list] = tmp.loc[tmp.bin == 1, col_list] * (1 - weight)

        # tmp = tmp.loc[tmp.bin==0]
        tmp = tmp.loc[tmp.bin.isin([0, 1])]
        #print(tmp.bin.value_counts())
        tmp = tmp.groupby('app_id').mean()

        # print(tmp.shape)
        # print(tmp.label.head(3))
        tmp.label = tmp.label.astype(int)
        # print(tmp.shape)
        score_list = accuracy(tmp)
        logger.info(f'weight:{weight}, score_list:{score_list}. base_train:{base_train}, File:{file}')
        total = score_list[1]
        score[weight] = total

    logger.info(f'Score list for file:{file}\n{score}')

    base_score = list(score.values())[-1]

    score = sorted(score.items(), key=lambda kv: kv[1])
    best_score = score[-1][-1]
    best_weight = score[-1][0]
    grow = best_score-base_score

    logger.info(f'====best_weight:{best_weight:3.2}, best_score:{best_score:6.5f}/{grow:6.5f},base_train:{base_train},File:{file}')
    return best_weight


def compare(file='./output/sub/80000_v36_07_bt_True_mean_top2_000_865700.csv'):
    df = pd.read_csv(file)

    from spider.mi import get_final_feature
    final = get_final_feature()

    df = pd.merge(final, df, how='left', on='id')

    def check(row):
        if len(str(row.type_id)) == 0:
            return None

        label_list = row.type_id.split('|')

        return str(row.label1) in label_list or str(row.label2) in label_list

    df['is_corr'] = df.apply(lambda row: check(row), axis=1)

    print(df.shape, '\n', df.is_corr.value_counts())
    df = df.loc[df.is_corr == False]

    type_name = get_app_type()
    type_name = type_name.set_index('type_id')
    type_name.index = type_name.index.astype(str)

    df.label1 = df.label1.astype(str).replace(type_name.to_dict()['type_name'])
    df.label2 = df.label2.astype(str).replace(type_name.to_dict()['type_name'])
    df.type_id = df.type_id.astype(str).replace(type_name.to_dict()['type_name'])

    print(df['from'].value_counts())

    return df

@timed()
@file_cache()
def get_oof_version(version, top, weight):


    with timed_bolck(f'Cal sub for top:{top}, weight:{weight:3.2f}, version:{version}'):
        for base_train in [True]:
            file_list = get_file_list(version, top)

            # file_list = file_list_1 +  file_list_2 + file_list_3 + file_list_4
            logger.info(f'File List {version} :{file_list}')

            res = get_feature_oof(file_list, weight, base_train)

    return res#, len(file_list)

@timed()
def main():
    oof_list  = []
    file_cnt = 0
    top = 4
    weight = 0
    oof_weight_list = []
    for version in ['v36', 'v43','v72','v73']:
        #version = get_args().version

        res = get_oof_version(version, top, weight)
        #Align all the porbiblity to 1


        train = res.loc[res.label != '0']
        score_list = accuracy(train)
        #oof_weight_list.append((score_list[1]))
        logger.info(f'Score for train{train.shape}/{res.shape}:{version}:{score_list}')

        res.iloc[:, :-1] =  res.iloc[:, :-1].apply(lambda row: row / row.sum(), axis=1)

        oof_list.append(res)


    oof = pd.concat(oof_list)

    oof.to_pickle(f'./output/{file_cnt:02}_{len(oof_list)}_tmp_res_val.pkl')

    label_raw = oof_list[0].label#.drop_duplicates()
    print('oof, final before=', oof.shape)
    oof = oof.groupby(oof.index).mean()
    #oof.iloc[:, :-1] = oof.iloc[:, :-1].apply(lambda row: row / row.sum(), axis=1)

    print('oof, final after=', oof.shape)
    oof['label'] = label_raw

    oof['label'] = oof['label'].fillna('0')

    res =  oof
    train = res.loc[res.label != '0']

    score_list = accuracy(train)
    total = score_list[1]
    logger.info(f'get the final score:{total} base on train:{train.shape}')

    ex_file = f'./output/{version}_bt_change_file_top{top}_w{weight}_{int(total * 10 ** 6):06}.csv'
    res.to_csv(ex_file)
    logger.info(f'Exchange file save to:{ex_file}')
    file_name = f'{version}_{file_cnt:02}_new_mean_top{top}_{int(weight * 100):03}_{int(total * 10 ** 6):06}.csv'
    res = gen_sub_file(res.loc[res.label == '0'], file_name)



if __name__== '__main__':
    FUNCTION_MAP = {'main': main,  }

    args = get_args()

    func = FUNCTION_MAP[args.command]
    func()



"""
nohup python -u ./core/ensemble_new.py  main  >> ensemble_newx.log 2>&1 &
"""



