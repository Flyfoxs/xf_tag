from sklearn.model_selection import StratifiedKFold

from core.feature import *


def get_split_group():
    apptype_train = pd.read_csv(f'{input_dir}/apptype_train.dat', sep='\t',
                                names=['app_id', 'type_id', 'app_des'],
                                quoting=3,
                                )

    apptype_train = apptype_train.sort_values('app_id')
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

    gp_list = list(folds.split(apptype_train, apptype_train.type_id.astype('category').cat.codes))

    train_list = [apptype_train.iloc[gp, 0].values for gp, _ in gp_list]

    val_list = [apptype_train.iloc[gp, 0].values for _, gp in gp_list]

    return train_list, val_list



def split_df_by_index(index, fold):
    app_id = pd.Series(index).apply(lambda val: val.split('_')[0])
    bin   = pd.Series(index).apply(lambda val: val.split('_')[-1]).astype(int)
    df = pd.concat([app_id,bin], axis=1)
    df.columns = ['app_id', 'bin']

    #print(df.shape, df.head)
    train_list, val_list = get_split_group()
    train_gp = train_list[fold]
    val_gp = val_list[fold]

    train_bin = list(range(get_args().max_bin))

    val_bin= [0,1,2]

    logger.info(f'The original bin_id distribution in val data set:\n{ df.loc[(df.app_id.isin(val_gp))].bin.value_counts() } ')

    return df.loc[(df.app_id.isin(train_gp)) & (df.bin.isin(train_bin))].index.values, \
           df.loc[(df.app_id.isin(val_gp)) &   (df.bin.isin(val_bin))].index.values

