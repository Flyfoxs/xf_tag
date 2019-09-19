import os
import sys

import numpy as np

from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI



'''Can be found at https://github.com/ymcui/Chinese-PreTrained-XLNet'''
checkpoint_path = '/users/hdpsbp/HadoopDir/felix/xlnet'
vocab_path = os.path.join(checkpoint_path, 'spiece.model')
config_path = os.path.join(checkpoint_path, 'xlnet_config.json')
model_path = os.path.join(checkpoint_path, 'xlnet_model.ckpt')

# Tokenize inputs
tokenizer = Tokenizer(vocab_path)
text = "给岁月以文明"
tokens = tokenizer.encode(text)

