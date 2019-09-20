#!/bin/bash

PYTHONPATH=/users/hdpsbp/bk/df_jf:/users/hdpsbp/felix/keras:$PYTHONPATH

PATH=/apps/dslab/anaconda/python3/bin:$PATH



#for level in 0.9 1 1.1 1.2 1.4 1.5 0.8
#for level in  1.4 1.5 0.8
for level in  0.75 0.85 0.7
do
    #echo python -u  core/train.py train_ex {} [] {0:$level, 3:$level, 4:$level, 6:$level, 9:$level} #> log/search_$level.log 2>&1
    python -u  core/train.py train_ex {} [] \{0:$level,3:$level,4:$level,6:$level,9:$level\}  > log/search2_$level.log 2>&1
done
