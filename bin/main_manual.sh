#!/bin/bash
cd "$(dirname "$0")"
cd ..

for i in {1..100};
do
    echo $i
    python -u ./core/bert_manual.py train_base  >> manual_batch_bin_0.log 2>&1
done




