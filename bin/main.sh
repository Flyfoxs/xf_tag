#!/bin/bash
cd "$(dirname "$0")"
cd ..

for i in {1..100};
do
    for fold in {0..4};
    do
        python -u ./core/bert.py --fold ${fold} train_base  >> batch_fold_${fold}.log 2>&1
    done
done




