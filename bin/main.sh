#!/bin/bash
cd "$(dirname "$0")"
cd ..

##for i in {1..10};
#for i in $(seq 0 $1)

#if not input $1, default value is 100
for i in $(seq 0 ${1:-100})
do
    for fold in {0..4};
    do
        python -u ./core/bert.py --fold=${fold} train_base  >> fold_${fold}_"$(hostname)".log 2>&1

    done
done




