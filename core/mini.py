
from file_cache.utils.util_log import *

vector_size = 200

def gen_mini_embedding(wv_from_text, word_list):
    from multiprocessing.dummy import Pool

    from functools import partial

    partition_num = 8
    import math
    partition_length = math.ceil(len(word_list)/partition_num)

    partition_list = [ word_list[i:i+partition_length]  for i in range(0, len(word_list), partition_length )]
    logger.debug(f'The word list split to {len(partition_list)} partitions:{[ len(partition) for partition in partition_list]}')
    thread_pool = Pool(processes=partition_num)
    process = partial(gen_mini_partition,wv_from_text=wv_from_text )

    wv_list = thread_pool.map(process, partition_list)
    thread_pool.close(); thread_pool.join()

    del wv_from_text

    return pd.concat(wv_list)


def compute_ngrams(word, min_n, max_n):
    # BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
    extended_word = word
    ngrams = []
    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    res =  list(set(ngrams))
    return res

def wordVec(word,wv_from_text:dict,min_n = 1, max_n = 3):
    '''
    ngrams_single/ngrams_more,主要是为了当出现oov的情况下,最好先不考虑单字词向量
    '''

    # 如果在词典之中，直接返回词向量
    if word in wv_from_text.index:
        return wv_from_text.loc[word]
    else:
        word_size = vector_size
        # 计算word的ngrams词组
        ngrams = compute_ngrams(word,min_n = min_n, max_n = max_n)
        # 不在词典的情况下
        word_vec = np.zeros(word_size, dtype=np.float32)
        ngrams_found = 0
        ngrams_single = [ng for ng in ngrams if len(ng) == 1]
        ngrams_more = [ng for ng in ngrams if len(ng) > 1]
        # 先只接受2个单词长度以上的词向量
        for ngram in ngrams_more:
            if ngram in wv_from_text.index:
                word_vec += wv_from_text.loc[ngram]
                ngrams_found += 1
                #print(ngram)
        # 如果，没有匹配到，那么最后是考虑单个词向量
        if ngrams_found == 0:
            for ngram in ngrams_single:
                if ngram in wv_from_text.index:
                    word_vec += wv_from_text.loc[ngram]
                    ngrams_found += 1
                elif ngram.lower() in wv_from_text.index:
                    word_vec += wv_from_text.loc[ngram.lower()]
                    ngrams_found += 1
                else:
                    logger.warning(f'Can not find {ngram} in wv')
        if ngrams_found > 0:
            return word_vec / max(1, ngrams_found)
        else:
            logger.error('all ngrams for word "%s" absent from model' % word)
            return None

@timed()
def gen_mini_partition(word_set, wv_from_text):

    mini = pd.DataFrame(np.zeros((len(word_set), vector_size)), index=word_set, )
    # for i in tqdm(range(len(word_set))):
    for i in range(len(word_set)):
        word = word_set[i]
        vector = wordVec(word, wv_from_text, 1, 3)
        if vector is not None:
            mini.loc[word] = vector
        else:
            logger.debug(f'Can not find vec for:{len(word)},{word}')
            mini.loc[word] = np.zeros(vector_size)

    return mini

@timed()
def gen_tx_mini():
    word2vec_tx, vector_size = './input/Tencent_AILab_ChineseEmbedding.txt', 200

    from core.feature import load_embedding, get_word_cnt

    embed = load_embedding(word2vec_tx)
    word_list = get_word_cnt()
    logger.info(word_list[:5])
    data = gen_mini_embedding(embed, word_list.word.values)

    logger.debug(f'The length of the vector is {data.shape}')

    fname = "./input/mini_tx.kv"
    np.savetxt(fname, data.reset_index().values,
               delimiter=" ",
               header="{} {}".format(len(data), len(data.columns)),
               comments="",
               fmt=["%s"] + ["%.6f"] * len(data.columns))

    logger.info(f'Mini dict save to {fname}')

if __name__ == '__main__':
   from fire import Fire
   Fire()

   """
   nohup python -u core/mini.py gen_tx_mini > mini.log 2>&1 &
   """