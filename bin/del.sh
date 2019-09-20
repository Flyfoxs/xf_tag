#!/usr/bin/env bash
cd "$(dirname "$0")"
cd ..



for fold in 2 2 2
do
    python -u ./core/bert.py --fold=${fold} train_base  >> fold_${fold}_"$(hostname)".log 2>&1
done

