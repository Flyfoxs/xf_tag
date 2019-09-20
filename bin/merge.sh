#!/bin/bash

#PYTHONPATH=/users/hdpsbp/HadoopDir/felix/df_jf:/users/hdpsbp/felix/keras:$PYTHONPATH

#PATH=/apps/dslab/anaconda/python3/bin:$PATH



#rm -rf ./output/blocks/*.csv

python ./core/merge.py > merge.log 2>&1
