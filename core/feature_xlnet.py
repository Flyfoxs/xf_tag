
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

def get_app_type():
    apptype = pd.read_csv(f'{input_dir}/apptype_id_name.txt', delimiter='\t', quoting=3, names=['type_id', 'type_name'],
                          dtype=type_dict)
    return apptype

@timed()
def get_raw_data():
    apptype_train = extend_train_set()

    apptype = get_app_type()

    apptype_test = pd.read_csv(f'{input_dir}/app_desc.dat', delimiter='\t', quoting=3, header=None, names=['app_id', 'app_des'], )
    #apptype_test.index = apptype_test.app_id
    apptype_test['app_id_ex'] = apptype_test.app_id

    data = pd.concat([apptype_test, apptype_train], axis=0)
    from spider.mi import filter_desc
    data.app_des = data.app_des.apply(lambda val: filter_desc(val))

    data = pd.merge(data, apptype, on='type_id', how='left')

    data['len_'] = data.app_des.apply(lambda val: len(val))

    return data.sort_values(['type_cnt'])

@lru_cache()
def get_tokenizer():
    from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI

    '''Can be found at https://github.com/ymcui/Chinese-PreTrained-XLNet'''
    checkpoint_path = xlnet_path
    vocab_path = os.path.join(checkpoint_path, 'spiece.model')
    # config_path = os.path.join(checkpoint_path, 'xlnet_config.json')
    # model_path = os.path.join(checkpoint_path, 'xlnet_model.ckpt')


    # Tokenize inputs
    tokenizer = Tokenizer(vocab_path)
    return tokenizer



def get_ids_from_text(text):
    def convert_tokens_to_ids(tokens):
        tokenizer = get_tokenizer()
        token_ids = tokenizer.encode(tokens)
        return token_ids
    ids = convert_tokens_to_ids(text)[1:]

    return [len(ids), ','.join([str(id) for id in ids])]


@timed()
def get_app_des_2_ids(data):
    data = data.drop_duplicates(['app_id_ex'])
    with timed_bolck(f'str to bert format for DF:{data.shape}'):
        # On app_id have multiply type_id
        ids = np.array(list([get_ids_from_text(text) for text in data.app_des.values.tolist()]))

    data['ids_lens_total'] = ids[:, 0].astype(int)

    data['ids'] = ids[:, 1]

    return data


@timed()
def split_app_des(df, split_len):
    if 'app_des' in df.columns:
        del df['app_des']
    seq_len = split_len - 2

    df_list = []

    def split_ids(ids, seq_len, window, i):
        from keras_xlnet import tokenizer
        ids = ids.split(',')

        encoded = ['19'] + ids[i * window : i * window + seq_len]
        ids = [str(tokenizer.Tokenizer.SYM_PAD)] * (seq_len+1 - len(encoded))  + encoded + [str(tokenizer.Tokenizer.SYM_CLS)]
        return pd.Series({'ids':','.join(ids), 'ids_lens_bin': len(ids)})


    window = get_args().seq_len - 2
    for i in tqdm(range(4), desc=f'split app des, window:{window}, seq_len:{seq_len}'):
        tmp = df.loc[(df.ids_lens_total > i * window)].copy()
        if len(tmp)==0:
            break
        tmp[['ids', 'ids_lens_bin']] = tmp.ids.apply(lambda val: split_ids(val, seq_len, window, i))
        tmp['bin'] = i
        tmp['app_id_ex_bin'] = tmp.app_id_ex + '_' + str(i)

        #logger.info(f'\nThere are {len(tmp)} records between [{i*seq_len},  {(i+1)*seq_len}) need to split.')

        df_list.append(tmp)

    logger.info(f'DF#{df.shape} split to {i + 1} groups, with seq_len_input={seq_len}, seq_len = {seq_len}')
    logger.info(f'The split result is: { [len(df)  for df in df_list] }')
    return pd.concat(df_list, axis=0)



@lru_cache()
@timed()
def get_label_id():
    app_type = pd.read_csv(f'{input_dir}/apptype_id_name.txt', delimiter='\t', quoting=3, names=['type_id', 'type_name'],
                          dtype=type_dict)
    app_type = app_type.loc[app_type.type_id.str.len()>=6]
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
def accuracy(res, topn=2, msg=''):

    if res is None or len(res) == 0:
        raise Exception(f'input is none')

    res = res.fillna(0)

    res = res.loc[res.label.astype(float) > 0].copy()
    logger.info(f'Accuracy base on res:{res.shape}')

    y = res.loc[:, 'label'].copy().astype(int)  # .astype(str)

    _, id2label = get_label_id()

    y = y.replace(id2label)
    # logger.info(f'Y=\n{y.head()}')

    res_list = np.zeros(topn)
    for i in tqdm(range(1, topn + 1, 1), desc=f'cal acc#{len(res)} for label(+),{msg}'):
        res[f'label{i}'] = res.loc[:, id2label.values()].idxmax(axis=1)  # .values
        # Exclude top#1

        for index, col in res[f'label{i}'].items():
            # logger.info(f'top#1 is {index}, {col}')
            res.loc[index, col] = np.nan
        res_list[i - 1] = sum(res[f'label{i}'].values.astype(int) == y.values.astype(int)) / len(res)

    score = np.round(np.cumsum(res_list),4)
    out_msg = f'\nBase on {res.shape} get {score}, msg:{msg}'
    logger.info(out_msg)
    #print(out_msg)
    return score

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

class Bert_Embed():


    @staticmethod
    def get_embed_wordvec_file():
        fname = bert_wv
        if os.path.exists(fname):
            return fname
        else:
            type_id = Bert_Embed._get_embed_from_type_name()

            app_type = get_app_type()
            app_type = app_type.drop_duplicates('type_name')
            app_type = app_type.set_index('type_id')

            type_name =type_id.copy()
            type_name = pd.merge(type_name, app_type, how='right', left_index=True, right_index=True)
            type_name = type_name.set_index('type_name')

            type_all = pd.concat([type_id, type_name])
            #del type_all['type_id']

            app_desc = Bert_Embed._get_embed_from_app_desc()

            data = pd.concat([type_all, app_desc])



            with timed_bolck(f'Save data#{data.shape} records to :{fname}'):
                np.savetxt(fname, data.reset_index().values,
                           delimiter=" ",
                           header="{} {}".format(len(data), len(data.columns)),
                           comments="",
                           fmt=["%s"] + ["%.6f"] * len(data.columns))

            return fname




    @file_cache()
    @staticmethod
    def _get_embed_from_app_desc():
        os.environ['TF_KERAS'] = '1'
        X = get_feature_bert(get_args().seq_len)
        input1_col = [col for col in X.columns if str(col).startswith('bert_')]
        X = X.loc[:, input1_col]
        logger.info(f'X shape:{X.shape}')
        return Bert_Embed._get_embed_by_bert(X)



    @file_cache()
    @staticmethod
    def _get_embed_from_type_name():
        os.environ['TF_KERAS'] = '1'
        app_type = get_app_type()

        ids = app_type.type_name.apply(lambda text: '101,' + get_ids_from_text(text)[1] + ',102')

        df = pd.DataFrame(np.zeros((len(app_type), 128))).add_prefix('bert_')

        tmp = ids.str.split(',', expand=True).add_prefix('bert_').fillna(0).astype(int)

        df.loc[:, tmp.columns] = tmp
        df.index = app_type.type_id
        df = df.astype(int)
        logger.info(f'df shape:{df.shape}')
        return Bert_Embed._get_embed_by_bert(df)




    @staticmethod
    def _get_embed_by_bert(X):
        with timed_bolck(f'Prepare train model'):

            from keras_bert import load_trained_model_from_checkpoint

            model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, seq_len=get_args().seq_len, )
            #model.summary(line_length=120)

            from tensorflow.python import keras
            from keras_bert import AdamWarmup, calc_train_steps
            inputs = model.inputs[:2]
            dense = model.get_layer('NSP-Dense').output
            model = keras.models.Model(inputs, dense)#.summary()



        with timed_bolck(f'try to gen embed DF{len(X)}'):
            input1_col = [col for col in X.columns if str(col).startswith('bert_')]
            # train_x, train_y = filter_short_desc(train_x, train_y)

            input1 = X.loc[:, input1_col]  # .astype(np.float32)
            input2 = np.zeros_like(input1)  # .astype(np.int8)

            logger.info(f'NN Input1:{input1.shape}, Input2:{input2.shape}')

            label2id, id2label = get_label_id()
            from keras_bert import get_custom_objects
            import tensorflow as tf
            with tf.keras.utils.custom_object_scope(get_custom_objects()):
                res_list = []
                partition_len = 5000
                for sn in tqdm(range(1 + len(X) // partition_len), 'gen embeding'):
                    tmp = X.iloc[sn * partition_len: (sn + 1) * partition_len]
                    # print('\nbegin tmp\n', tmp.iloc[:3,:3].head())
                    res = model.predict([tmp.loc[:, input1_col], np.zeros_like(tmp.loc[:, input1_col])])
                    res = pd.DataFrame(res, index=tmp.index).add_prefix('embd_bert')
                    # print('\nend tmp\n', res.iloc[:3, :3].head())
                    res_list.append(res)

                res = pd.concat(res_list)

        return res


@lru_cache()
def get_args():
    parser = argparse.ArgumentParser()
    # parser.set_defaults(func=test_abc)

    from random import randrange


    parser.add_argument("--fold", type=int, default=0, help="Split fold")
    parser.add_argument("--max_bin", type=int, default=2, help="How many bin need to train")
    parser.add_argument("--min_len_ratio", type=float, default=0.8, help="The generated sample seq less than min_len will be drop")
    parser.add_argument("--epochs", type=int, default=16, help="How many epoch is need, default is 2 or 3")
    parser.add_argument("--frac", type=float, default=1, help="How many sample will pick")


    #parser.add_argument("--window", type=int, default=local_seq_len-2, help="Rolling to gen sample for training")
    parser.add_argument("--cut_ratio", type=float, default=0.1, help="Reduce the end of the desc")
    parser.add_argument("--max_cut_num", type=int, default=30, help="Reduce the end of the desc")

    parser.add_argument("--seq_len", type=int, default=128, help="How many words for bert")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--version", type=str, default='v74', help="version")

    parser.add_argument('--trial', type=int, default=0)

    parser.add_argument('command', type=str, default='cmd')

    args = parser.parse_args()
    logger.info(f'Args===,{args}')
    return args


@timed()
@file_cache(prefix=get_args().version)
def get_feature_xlnet(seq_len):

    raw = get_raw_data()
    #print('====0', len(raw.loc[raw.app_id == 'BA915EC5E4CB0884C08C8DD9E9F1FD8F']))
    data = get_app_des_2_ids(raw)
    #print('====1', len(data.loc[data.app_id=='BA915EC5E4CB0884C08C8DD9E9F1FD8F']))
    data = split_app_des(data, seq_len)

    bert = data.ids.str.split(',', expand=True).add_prefix('bert_').fillna(0).astype(int)

    with timed_bolck(f'Join bert#{bert.shape} and raw#{raw.shape} data'):
        old_shape = bert.shape

        bert['app_id'] = data.app_id.values
        bert['app_id_ex'] = data.app_id_ex.values
        bert['app_id_ex_bin'] = data.app_id_ex_bin.values

        bert['ids_lens_bin'] = data.ids_lens_bin.values
        bert['ids_lens_total'] = data.ids_lens_total.values

        bert['bin'] = data.bin.values
        if 'app_des' in raw: del raw['app_des']
        del raw['app_id']
        bert = pd.merge(bert, raw, how='left', on=['app_id_ex'])

        bert.index = bert.app_id_ex_bin
        logger.info(f'Merge extend shape from {old_shape}, {raw.shape} to {bert.shape}')

    padding_analysis = bert.loc[:, f'bert_{seq_len-1}'].value_counts().sort_index()
    logger.info(f'padding_analysis(bert_{seq_len-1}):\n{padding_analysis}')
    bert.index.name = 'index'
    return bert.sort_values(['app_id_ex_bin'], ascending=False)


def get_feature_type_des(seq_len):
    type_name = get_app_type()
    ids = np.array(list([get_ids_from_text(text) for text in type_name.type_name.values.tolist()]))

    ids = pd.Series(ids[:, 1])
    ids = '101,' + ids + ',102'

    tmp = ids.str.split(',', expand=True).add_prefix('fea_type_').fillna(0).astype(int)
    tmp.index = type_name.index
    type_name[tmp.columns] = tmp

    for i in range(seq_len):
        if f'fea_type_{i}' not in type_name:
            type_name[f'fea_type_{i}'] = 0

    return type_name
#
# @timed()
# @file_cache()
# def get_feature_bert_wv():
#     with timed_bolck(f'Read wv by gensim'):
#         fname = Bert_Embed.get_embed_wordvec_file()
#         import gensim
#         word_vectors = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=False)
#         raw_bert = get_feature_bert(get_args().seq_len)
#         label2id, id2label = get_label_id()
#         df = pd.DataFrame(np.zeros((len(raw_bert), num_classes)), columns=label2id.keys(), index=raw_bert.index)
#
#
#     for col in tqdm(df.columns, desc=f'Cal distanc for DF:{df.shape}'):
#         df[col] = pd.Series(df.index).apply(lambda id_ex_bin: word_vectors.distance(id_ex_bin, col) ).values
#
#     return df

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

#
# @timed()
# def get_feature_bin(seq_len):
#     sample_bin = get_sample_bin()
#
#     app_des = get_feature_bert(seq_len)
#     select_col = [ col for col in app_des.columns if col.startswith('bert_')]
#     select_col = ['app_id'] + select_col
#     app_des = app_des.loc[app_des.bin==0].loc[:, select_col]
#
#     type_des = get_feature_type_des(seq_len)
#
#     sample_bin = pd.merge(sample_bin, app_des, how='left', on='app_id')
#     print(list(sample_bin.columns))
#     sample_bin = pd.merge(sample_bin, type_des, how='left', on='type_id')
#
#     sample_bin['app_id_bin'] = sample_bin.app_id + '_' + sample_bin.sn.astype(str)
#     sample_bin.index = sample_bin.app_id_bin
#
#     return sample_bin
#
# @timed()
# @file_cache()
# def get_sample_bin():
#     """
#     app_id_ex0, app_id_ex_bin,  type_id, predict_rank, predict_rank_ex,
#     :return:
#     """
#     from core.ensemble import get_feature_oof, gen_sub_file
#     top_file = 2
#     oof = get_feature_oof(version, top=top_file, weight=0)
#
#     # oof = oof.sample(frac=0.05)
#     # Make sure the real label put in label#1 column
#     for index, col in oof.label.items():
#         oof.loc[index, str(col)] = 1
#
#     top_label = 5
#     sub = gen_sub_file(oof, None, topn=top_label)
#
#     sub.index.name = 'app_id'
#     sub = sub.reset_index()
#
#     sample_list = []
#     for i in range(1, top_label + 1):
#         col_list = ['app_id', f'label{i}', f'label']
#         tmp = sub.loc[:, col_list].copy()
#         tmp.rename({f'label{i}': 'type_id'}, axis=1, inplace=True)
#
#         tmp['sn'] = i
#
#         tmp.label = tmp.label.astype(int)
#
#         # Test Sample
#         tmp.loc[tmp.label == 0, 'label'] = -1
#         if i == 1:
#             tmp.loc[tmp.label > 0, 'label'] = 1
#         else:
#             tmp.loc[tmp.label > 0, 'label'] = 0
#
#         sample_list.append(tmp)
#     res = pd.concat(sample_list)
#
#     res['app_id_bin'] = res.app_id + '_' + res.sn.astype(str)
#     return res.sort_values(['sn', 'app_id'])
#


if __name__ == '__main__':

    FUNCTION_MAP = {
                    'get_feature_bert_wv':get_feature_bert_wv,
                    'get_feature_bert':get_feature_bert,
                    #'get_feature_bin':get_feature_bin
                     }

    args = get_args()

    func = FUNCTION_MAP[args.command]
    func()

"""
nohup python -u core/feature.py get_feature_bert > feature.log 2>&1 &

nohup python -u core/feature.py get_feature_bin > feature.log 2>&1 &

 

"""