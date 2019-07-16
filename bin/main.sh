#!/bin/bash
cd "$(dirname "$0")"
cd ..

for i in {1..100};
do
    echo $i
    python -u ./core/bert.py train_base  >> batch_bin_1.log 2>&1
done




