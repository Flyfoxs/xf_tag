
import sys

import argparse

sys.path.append('./')

print(sys.path)

import json
import warnings
from functools import lru_cache
from core.conf import *



from deprecated import deprecated
from file_cache.utils.util_log import timed_bolck
from file_cache.cache import file_cache
from file_cache.utils.reduce_mem import *
from file_cache.utils.util_pandas import *
from sklearn.decomposition import TruncatedSVD

from tqdm import tqdm

@timed()
def extend_train_set():
    @timed()
    def split_type_id(df, type_sn):
        df_tmp = df.copy()
        df_tmp.type_id = df_tmp.type_id.apply(lambda val: val.split('|')[type_sn]
        if type_sn + 1 <= len(val.split('|')) else None)

        return df_tmp.loc[pd.notna(df_tmp.type_id)]

    apptype_train = pd.read_csv(f'{input_dir}/apptype_train.dat', sep='\t',
                                names=['app_id', 'type_id', 'app_des'],
                                quoting=3,
                                )

    apptype_train['type_cnt'] = apptype_train.type_id.apply(lambda val: len(val.split('|')))

    apptype_train_main = apptype_train.loc[apptype_train.type_cnt == 1]
    todo = apptype_train.loc[apptype_train.type_cnt > 1]

    logger.info(f'There are {len(todo)} recoreds have multiply type')

    split_res_list = []
    for type_sn in range(apptype_train['type_cnt'].max()):
        split_res_list.append(split_type_id(todo, type_sn))
    split_res = pd.concat(split_res_list)

    apptype_train = pd.concat([apptype_train_main, split_res])
    # apptype_train.type_id = apptype_train.type_id.astype(str)
    return apptype_train


@timed()
def get_data():
    apptype_train = extend_train_set()

    apptype = pd.read_csv(f'{input_dir}/apptype_id_name.txt', delimiter='\t', quoting=3, names=['type_id', 'type_name'],
                          dtype=type_dict)

    apptype_test = pd.read_csv(f'{input_dir}/app_desc.dat', delimiter='\t', quoting=3, header=None, names=['app_id', 'app_des'], )

    data = pd.concat([apptype_test, apptype_train], axis=0)
    data = pd.merge(data, apptype, on='type_id', how='left')
    return data


@lru_cache()
@timed()
def get_jieba():
    import jieba
    jieba.load_userdict('./input/jieba.txt')
    return jieba


@timed()
def get_similar_top(word_list: pd.Series, topn=3):
    wm = load_embedding_gensim(word2vec_tx)
    s = word_list.apply(
        lambda word: ','.join([k for k, v in wm.most_similar(word, topn=3)]) if word is not None and word in wm else '')
    return s.str.split(',', expand=True).add_prefix(f'{word_list.name}_similar_')


@timed()
@file_cache()
def get_app_type_ex():
    apptype = pd.read_csv(f'{input_dir}/apptype_id_name.txt', delimiter='\t', quoting=3, names=['type_id', 'type_name'],
                          dtype=type_dict)
    print(apptype[['type_name']].shape)
    split_words = get_split_words(apptype[['type_name']])

    split_words = split_words.jieba_txt.apply(lambda val: ','.join(val)).str.split(',', expand=True).add_prefix(
        'type_name_')
    # If can not split to 2 words, then don't split, keep both is None
    split_words.iloc[:, 0] = split_words.apply(lambda row: None if pd.isna(row[1]) else row[0], axis=1)

    apptype = pd.concat([apptype, split_words], axis=1)

    apptype['type_cnt'] = apptype.type_name_1.apply(lambda val: 0 if pd.isna(val) else 1)

    df_list = [apptype]
    for col in ['type_name', 'type_name_0', 'type_name_1']:
        similar = get_similar_top(apptype[col], topn=3)

        df_list.append(similar)

    return pd.concat(df_list, axis=1)




@timed()
def get_split_words(train):
    jieba = get_jieba()

    # train = get_data()
    # train = train.sort_values('type_cnt', ascending=False)
    # # train.jieba_txt = train.type_name.fillna(',')
    #

    train['jieba_txt'] = ''
    for col in train :
        if col != 'jieba_txt':
            train['jieba_txt'] = train['jieba_txt'] + ',' + train.loc[:, col].fillna('')

    jieba_txt = train.jieba_txt.apply(lambda text: list(jieba.cut(text, cut_all=False)))
    jieba_txt = jieba_txt.apply(lambda text: [word for word in text if word not in [' ', ',', ')', '(']])

    jieba_txt = jieba_txt.to_frame()
    #print(jieba_txt.head())
    # print(jieba_txt.shape, jieba_txt.head())
    jieba_txt['jieba_len'] = jieba_txt.jieba_txt.apply(lambda text: len(text))
    return jieba_txt


@timed()
@file_cache()
def get_word_cnt(col_list=['type_name', 'app_des']):

    data = get_data()
    jieba_txt= get_split_words(data[col_list])

    import collections
    count = collections.Counter()

    for text in tqdm(jieba_txt['jieba_txt'].values, desc="split to count"):
        for word in text:
            count[word.lower()] += 1
    return pd.DataFrame({'word':list(count.keys()), 'count':list(count.values())})


def load_embedding(path_txt, type='gensim'):
    if type=='gensim':
        return load_embedding_gensim(path_txt)
    else:
        return load_embedding_txt(path_txt)

@lru_cache()
@timed()
def load_embedding_txt(path_txt):
    embedding_index = {}
    f = open(path_txt, encoding='utf8')  # .read().split('\r\n')
    for index, line in enumerate(f):
        values = line.split(' ')

        if index == 0:
            dim = int(values[1])
            logger.info(f'The word2vec dim is :{dim}, {path_txt}')
            continue

        word = values[0]
        # print(word, len(values), values[1:dim+1])
        coefs = np.asarray(values[1:dim + 1], dtype='float32')
        embedding_index[word] = coefs
#         if index == 2:
#             break
    f.close()
    df = pd.DataFrame(list(embedding_index.values()), index = list(embedding_index.keys()))
    df.index.name='word'
    return df#.reset_index()

@lru_cache()
@timed()
def load_embedding_gensim(path_txt):
    from gensim.models import KeyedVectors
    wv_from_text = KeyedVectors.load_word2vec_format(path_txt, binary=False)
    return wv_from_text

@timed()
@lru_cache()
def check_word_exist(path_txt='./input/mini_tx.kv'):
    word_df = get_word_cnt()
    embed = load_embedding(path_txt)
    word_df['exist'] = word_df.word.apply(lambda val: 1 if val in embed.index.values else 0)

    return word_df



if __name__ == '__main__':
    app_type = get_app_type_ex().sort_values('type_cnt')
    app_type.head()

"""

"""