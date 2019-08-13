from random import randrange

input_dir = './input/zip/'

type_dict = {
    'type_id': 'str',

}

word2vec_tx, vector_size = './input/Tencent_AILab_ChineseEmbedding.txt',  200

word2vec_tx_mini = './input/mini_tx.kv'

num_classes = 126  #get_label_id()

oof_prefix = 'v21'


bert_wv = "./input/bert.kv"
####Bert Config
import os
SEQ_LEN=128#-randrange(0, 5)*8
pretrained_path = './input/model/chinese_L-12_H-768_A-12'
#pretrained_path = './input/model/chinese_wwm_ext_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

#######