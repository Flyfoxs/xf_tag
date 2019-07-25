
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

    apptype_train.index = apptype_train.app_id
    apptype_train['app_id_ex'] = apptype_train.app_id

    apptype_train['type_cnt'] = apptype_train.type_id.apply(lambda val: len(val.split('|')))

    apptype_train_main = apptype_train.loc[apptype_train.type_cnt == 1]
    todo = apptype_train.loc[apptype_train.type_cnt > 1]

    logger.info(f'There are {len(todo)} recoreds have multiply type')

    split_res_list = []
    for type_sn in range(apptype_train['type_cnt'].max()):
        tmp=split_type_id(todo, type_sn)
        tmp['app_id_ex'] = tmp.app_id_ex + f'_{type_sn}'
        split_res_list.append(tmp)
    split_res = pd.concat(split_res_list)

    apptype_train = pd.concat([apptype_train_main, split_res])
    # apptype_train.type_id = apptype_train.type_id.astype(str)
    return apptype_train

@timed()
def get_raw_data():
    apptype_train = extend_train_set()

    apptype = pd.read_csv(f'{input_dir}/apptype_id_name.txt', delimiter='\t', quoting=3, names=['type_id', 'type_name'],
                          dtype=type_dict)

    apptype_test = pd.read_csv(f'{input_dir}/app_desc.dat', delimiter='\t', quoting=3, header=None, names=['app_id', 'app_des'], )
    #apptype_test.index = apptype_test.app_id
    apptype_test['app_id_ex'] = apptype_test.app_id

    data = pd.concat([apptype_test, apptype_train], axis=0)
    data = pd.merge(data, apptype, on='type_id', how='left')

    data['len_'] = data.app_des.apply(lambda val: len(val))

    return data.sort_values(['type_cnt'])


@timed()
def get_app_des_2_ids(data):
    import codecs
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)



    from keras_bert import Tokenizer
    tokenizer = Tokenizer(token_dict)
    def convert_tokens_to_ids(tokens):
        tokens = tokenizer._tokenize(tokens)
        token_ids = tokenizer._convert_tokens_to_ids(tokens)
        return token_ids

    def get_ids_from_text(text):
        ids = convert_tokens_to_ids(text)
        return [len(ids), ','.join([str(id) for id in ids])]

    with timed_bolck('str to bert format'):
        # On app_id have multiply type_id
        data = data.drop_duplicates(['app_id'])
        ids = np.array(list([get_ids_from_text(text) for text in data.app_des.values.tolist()]))

    data['ids_lens'] = ids[:, 0].astype(int)
    data['ids_lens_total'] = data['ids_lens']

    data['ids'] = ids[:, 1]

    return data


@timed()
def split_app_des(df, split_len=SEQ_LEN):
    if 'app_des' in df.columns:
        del df['app_des']
    seq_len = split_len - 2

    df_list = []

    def split_ids(ids, seq_len, i):
        ids = ids.split(',')
        ids = ['101'] + ids[i * seq_len : (i + 1) * seq_len] + ['102']
        return pd.Series({'ids':','.join(ids), 'ids_lens': len(ids)})


    for i in tqdm(range(4), desc='split app des'):
        tmp = df.loc[(df.ids_lens > i * seq_len)].copy()
        if len(tmp)==0:
            break
        tmp[['ids', 'ids_lens']] = tmp.ids.apply(lambda val: split_ids(val, seq_len, i))
        tmp['bin'] = i
        tmp['app_id_ex_bin'] = tmp.app_id_ex + '_' + str(i)

        #logger.info(f'\nThere are {len(tmp)} records between [{i*seq_len},  {(i+1)*seq_len}) need to split.')

        df_list.append(tmp)

    logger.info(f'DF#{df.shape} split to {i + 1} groups, with seq_len_input={seq_len}, seq_len = {seq_len}')
    logger.info(f'The split result is: { [len(df)  for df in df_list] }')
    return pd.concat(df_list, axis=0)


@lru_cache()
def get_label_id():
    app_type = pd.read_csv(f'{input_dir}/apptype_id_name.txt', delimiter='\t', quoting=3, names=['type_id', 'type_name'],
                          dtype=type_dict)
    labels = app_type.type_id.values.tolist()
    # Construction of label2id and id2label dicts
    label2id = {l: i for i, l in enumerate(set(labels))}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label

@timed()
def get_data():
    raw_data = get_raw_data()

    manual = get_feature_manual(10)

    seq = get_feature_seq_input_sentences()

    data = pd.concat([raw_data, seq, manual, ], axis=1)

    logger.info(f'Shape of data:{data.shape}, raw_data:{raw_data.shape} manual:{manual.shape}, seq:{seq.shape} ')

    return data.sort_values(['type_cnt'])


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
        lambda word: ','.join([k for k, v in wm.most_similar(word, topn=topn)]) if word is not None and word in wm else '')
    return s.str.split(',', expand=True).add_prefix(f'{word_list.name}_similar_')


@timed()
@file_cache()
def get_key_word_list(similar_cnt=10):
    df  = pd.read_csv('./input/type_list_jieba.txt', names=['key_words'], header=None).drop_duplicates()
    similar = get_similar_top(df.key_words, similar_cnt)
    return pd.concat([df, similar], axis=1)


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

    jieba_txt = train.jieba_txt.apply(lambda text: list(jieba.cut(text, cut_all=True)))
    jieba_txt = jieba_txt.apply(lambda text: [word for word in text if word not in [' ', ',', '(', ']', '[', ']']])

    jieba_txt = jieba_txt.to_frame()
    #print(jieba_txt.head())
    # print(jieba_txt.shape, jieba_txt.head())
    jieba_txt['jieba_len'] = jieba_txt.jieba_txt.apply(lambda text: len(text))
    return jieba_txt


@timed()
@file_cache()
def get_word_cnt():

    #Type Name
    import collections
    count = collections.Counter()
    word_list = get_key_word_list()
    word_list = pd.Series(word_list.values.reshape(1, -1)[0]).dropna().drop_duplicates()
    for word in tqdm(word_list, desc='type & similar'):
        count[word.lower()] += 1


    data = get_raw_data()
    jieba_txt= get_split_words(data[['app_des']])
    for text in tqdm(jieba_txt['jieba_txt'].values, desc="split docs"):
        for word in text:
            count[word.lower()] += 1
    return pd.DataFrame({'word':list(count.keys()), 'count':list(count.values())})


@timed()
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

# @timed()
# @lru_cache()
# def check_word_exist(path_txt='./input/mini_tx.kv'):
#     word_df = get_word_cnt()
#     embed = load_embedding(path_txt)
#     word_df['exist'] = word_df.word.apply(lambda val: 1 if val in embed.index.values else 0)
#
#     return word_df


@timed()
def accuracy(res):
    if res is None or len(res)==0:
        return 0, 0, 0

    res = res.loc[res.label.astype(float) > 0].copy()
    logger.info(f'Accuracy base on res:{res.shape}')

    y = res.loc[:,'label'].copy().astype(int)#.astype(str)

    _, id2label = get_label_id()

    y = y.replace(id2label)
    #logger.info(f'Y=\n{y.head()}')

    res['label1'] = res.iloc[:, :num_classes].idxmax(axis=1)#.values

    #Exclude top#1
    for index, col in res.label1.items():
        #logger.info(f'top#1 is {index}, {col}')
        res.loc[index, col] = np.nan

    res['label2'] = res.iloc[:, :num_classes].idxmax(axis=1)#.values

    acc1 = sum(res['label1'].values.astype(int) == y.values.astype(int)) / len(res)
    acc2 = sum(res['label2'].values.astype(int) == y.values.astype(int)) / len(res)

    return acc1, acc2, acc1+acc2



@timed()
@file_cache()
def get_feature_manual(n_topics):
    tfidf = get_feature_tfidf_type()
    #return tfidf
    lda = get_feature_lda(n_topics)

    manual = pd.concat([tfidf,lda], axis=1)

    manual['app_id'] = manual.index

    manual = manual.drop_duplicates(['app_id'])

    return manual


#@lru_cache()
@timed()
def get_feature_tfidf_type():
    #TODO need to analysis the missing, and manually split

    tfidf = get_tfidf_all()

    word_list = get_key_word_list()
    feature_name = pd.Series(word_list.values.reshape(1, -1)[0]).dropna().drop_duplicates()
    feature_missing = [col for col in feature_name if col not in tfidf.columns]
    #logger.warning(f'Feature missing#{len(feature_missing)}:{feature_missing}')

    feature_name = [col for col in feature_name if col in tfidf.columns]
    return tfidf.loc[:, feature_name].fillna(0).add_prefix('fea_tfidf_')

@timed()
#@file_cache()
@lru_cache()
def get_tfidf_all():
    with timed_bolck('Gen_ALL_docs'):
        data = get_raw_data()

        app_docs = get_split_words(data[['app_des']])

        docs = app_docs.jieba_txt.apply(lambda val: ','.join(val))

    with timed_bolck('CountVec'):

        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_df=0.85, stop_words=[',', '[', ']','(', ')'])

        word_count_vector = cv.fit_transform(docs)
        list(cv.vocabulary_.keys())[:10]
    with timed_bolck('TFIDF'):
        from sklearn.feature_extraction.text import TfidfTransformer

        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tf_idf_vector = tfidf_transformer.fit_transform(word_count_vector)

    with timed_bolck('Gen Sparse TFIDF'):
        df = pd.SparseDataFrame(tf_idf_vector, columns=cv.get_feature_names(), index=data.app_id)

    return df


@lru_cache()
@timed()
def get_word2id():

    word_id_vec =  load_embedding(word2vec_tx_mini, type='txt')

    word2id = {l: i for i, l in enumerate(set(word_id_vec.index.values))}

    return word2id

@file_cache()
@timed()
def get_feature_seq_input_sentences():
    data = get_raw_data()
    with timed_bolck('for df cut txt to words'):
        jieba = get_jieba()
        input_sentences = [list(jieba.cut(str(text), cut_all=True))  for text in data.app_des.values.tolist()]

    word2id = get_word2id()

    # Encode input words and labels
    X = [[word2id[word] for word in sentence if word in word2id] for sentence in input_sentences]
    max_words = 0 # maximum number of words in a sentence
    # Construction of word2id dict
    for sentence in input_sentences:
        if len(sentence) > max_words:
            max_words = len(sentence)
            # logger.debug(f'max_words={max_words}')
    logger.info(f'max_words={max_words}')

    with timed_bolck('X pad_sequences'):
        from keras.preprocessing.sequence import pad_sequences
        X = pad_sequences(X, max_words)

    return pd.DataFrame(X, index=data.app_id).add_prefix('seq_')


@timed()
@file_cache()
def get_feature_bert():

    raw = get_raw_data()
    data = get_app_des_2_ids(raw)
    data = split_app_des(data)

    bert = data.ids.str.split(',', expand=True).add_prefix('bert_').fillna(0).astype(int)


    with timed_bolck(f'Join bert#{bert.shape} and raw#{raw.shape} data'):
        old_shape = bert.shape

        bert['app_id'] = data.app_id.values
        bert['app_id_ex'] = data.app_id_ex.values
        bert['app_id_ex_bin'] = data.app_id_ex_bin.values
        bert['bin'] = data.bin.values
        bert['len_'] = data.len_.values
        if 'app_des' in raw: del raw['app_des']
        del bert['app_id_ex']
        del raw['len_']
        bert = pd.merge(bert, raw, how='left', on=['app_id'])

        bert.index = bert.app_id_ex_bin
        logger.info(f'Merge extend shape from {old_shape}, {raw.shape} to {bert.shape}')

    padding_analysis = bert.loc[:, f'bert_127'].value_counts().sort_index()
    logger.info(f'padding_analysis:\n{padding_analysis}')
    return bert.sort_values(['app_id_ex_bin'], ascending=False)


#6 hours
@timed()
@file_cache()
def get_feature_lda(n_topics):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    with timed_bolck('Gen_ALL_docs'):
        data = get_raw_data()

        app_docs = get_split_words(data[['app_des']])

        docs = app_docs.jieba_txt.apply(lambda val: ','.join(val))

    with timed_bolck('CountVec'):

        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_df=0.85, stop_words=[',', '[', ']','(', ')'])
        cntTf = cv.fit_transform(docs)

    with timed_bolck(f'Cal LDA#{n_topics}'):

        lda = LatentDirichletAllocation(n_components=n_topics,
                                        learning_offset=50.,
                                        random_state=666)
        docres = lda.fit_transform(cntTf)

    return pd.DataFrame(docres, columns=[f'fea_lda_{i}' for i in range(n_topics)], index=data.app_id)


def batch_manual():
    for n_topics in range(10, 100, 10):
        get_feature_manual(n_topics)


def get_args():
    parser = argparse.ArgumentParser()
    # parser.set_defaults(func=test_abc)

    from random import randrange
    parser.add_argument("--fold", type=int, default=0, help="Split fold")
    parser.add_argument("--max_bin", type=int, default=0, help="How many bin need to train")
    parser.add_argument("--min_len", type=int, default=100, help="The generated seq less than min_len will be drop")
    parser.add_argument("--epochs", type=int, default=randrange(2, 4), help="How many epoch is need, default is 2 or 3")
    parser.add_argument("--frac", type=float, default=1.0, help="How many sample will pick")

    subparsers = parser.add_subparsers()

    bert_parser = subparsers.add_parser('train_base')
    from core.bert import train_base
    bert_parser.set_defaults(func=train_base)

    manual = subparsers.add_parser('manual')
    from core.bert_manual import manual_train
    manual.set_defaults(func=manual_train)

    lgb_parser = subparsers.add_parser('train_ex')
    from core.lgb import train_ex
    lgb_parser.set_defaults(func=train_ex)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import fire
    fire.Fire()

"""
nohup python core/feature.py batch_manual > feature.log 2>&1 &

"""