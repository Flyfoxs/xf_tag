from sklearn.model_selection import StratifiedKFold

from core.feature import *


def get_split_group(random_state=2019):
    apptype_train = pd.read_csv(f'{input_dir}/apptype_train.dat', sep='\t',
                                names=['app_id', 'type_id', 'app_des'],
                                quoting=3,
                                )

    apptype_train = apptype_train.sort_values('app_id')
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    gp_list = list(folds.split(apptype_train, apptype_train.type_id.astype('category').cat.codes))

    train_list = [apptype_train.iloc[gp, 0].values for gp, _ in gp_list]

    val_list = [apptype_train.iloc[gp, 0].values for _, gp in gp_list]

    return train_list, val_list


@timed()
def split_df_by_index_no_bin(df, fold):
    #
    # sn = pd.Series(df.index).str[-1].astype(int)
    df = pd.Series(df.index).str[:32]



    train_list, val_list = get_split_group()
    train_gp = train_list[fold]
    val_gp = val_list[fold]

    return df.loc[(df.isin(train_gp))].index.values, \
           df.loc[(df.isin(val_gp)) ].index.values


def split_df_by_index(df, fold):
    index = df.index
    app_id = pd.Series(index).apply(lambda val: val.split('_')[0])
    bin   = pd.Series(index).apply(lambda val: val.split('_')[-1]).astype(int)
    df = pd.concat([app_id,bin], axis=1)
    df.columns = ['app_id', 'bin']

    #print(df.shape, df.head)
    train_list, val_list = get_split_group()
    train_gp = train_list[fold]
    val_gp = val_list[fold]

    train_bin = list(range(get_args().max_bin+1))

    val_bin= train_bin #[0,1]

    logger.info(f'split base on: train_bin:{train_bin}, val_bin:{val_bin}')
    logger.info(f'The original bin_id distribution in train data set:\n {df.loc[(df.app_id.isin(train_bin))].bin.value_counts()}')

    logger.info(f'The original bin_id distribution in val data set:\n{ df.loc[(df.app_id.isin(val_gp))].bin.value_counts() } ')

    return df.loc[(df.app_id.isin(train_gp)) & (df.bin.isin(train_bin))].index.values, \
           df.loc[(df.app_id.isin(val_gp)) &   (df.bin.isin(val_bin))].index.values


if __name__ == '__main__':
    for random in range(2019, 2099):
        train_list, val_list = get_split_group(random)
        gp = [len(val)  for val  in val_list]
        print(np.array(gp).std(), gp, random)
