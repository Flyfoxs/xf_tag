from random import randrange

input_dir = './input/zip/'

type_dict = {
    'type_id': 'str',

}

word2vec_tx, vector_size = './input/Tencent_AILab_ChineseEmbedding.txt',  200

word2vec_tx_mini = './input/mini_tx.kv'

num_classes = 126  #get_label_id()


bert_wv = "./input/bert.kv"
####Bert Config
import os
pretrained_path = '/users/hdpsbp/HadoopDir/felix/xf_tag/input/roebert' #'./input/model/chinese_L-12_H-768_A-12'

if not os.path.exists(pretrained_path):
    pretrained_path =  '/home/aladdin1/felix/robert'

#pretrained_path = './input/model/chinese_wwm_ext_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config_large.json')
checkpoint_path = os.path.join(pretrained_path, 'roberta_zh_large_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

check_type_list = ['stb', '50', '100', '200', '300','1000',
                   #'a2', 'a3', 'bd',
                    ]

#######

xlnet_path='/users/hdpsbp/HadoopDir/felix/xlnet'