# Initial

# 腾讯词向量下载地址

- https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz
ln -s ../../word_vec/Tencent_AILab_ChineseEmbedding.txt  Tencent_AILab_ChineseEmbedding.txt
- https://github.com/Embedding/Chinese-Word-Vectors
    Zhihu_QA 知乎问答

# 去除引号
sed -i 's/"//g' *.*


# 停用词
https://github.com/goto456/stopwords


# Type_id Cnt: 
152

# Bert

    wget -q https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    the download model will save to folder: input/model/chinese_L-12_H-768_A-12


# 输入目录

    input/
    ├── model
    │   ├── chinese_L-12_H-768_A-12
    │   │   ├── bert_config.json
    │   │   ├── bert_model.ckpt.data-00000-of-00001
    │   │   ├── bert_model.ckpt.index
    │   │   ├── bert_model.ckpt.meta
    │   │   └── vocab.txt
    │   └── uncased_L-12_H-768_A-12.zip
    ├── Tencent_AILab_ChineseEmbedding.txt
    └── zip
        ├── app_desc.dat
        ├── apptype_id_name.txt
        ├── apptype_train.dat
        └── mini.dat
