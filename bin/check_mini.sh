#!/bin/bash

PYTHONPATH=/users/hdpsbp/bk/df_jf:/users/hdpsbp/felix/keras:$PYTHONPATH

PATH=/apps/dslab/anaconda/python3/bin:$PATH



for bin_count in 8 #20
do
    echo $bin_count
    python ./core/check.py -L   --bin_count $bin_count  --gp_name lr_bin_$bin_count     \
                    > log/bin_"$(hostname)"_$bin_count.log 2>&1
done
# nohup ./bin/check_mini.sh 5 &
