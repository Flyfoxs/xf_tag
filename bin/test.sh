#!/bin/bash
cd "$(dirname "$0")"
cd ..

#for xx in {0..2};
#do
#    python -u  spider/mi.py bd  > bd.log 2>&1
#    python -u  spider/mi.py wdj  > wdj.log 2>&1
#    python -u  spider/mi.py xm  >  xm.log 2>&1
#
#    python -u  spider/mi.py tx_pkg  >  tx_pkg.log 2>&1
#    python -u  spider/mi.py tx_name  >  tx_name.log 2>&1
#
#    python spider/gen_file.py > gen_file.log 2>&1
#done

for xx in {0..2};
do
    for fold in 4 3 2 1 0
    do
        python -u ./core/bert.py --fold=${fold} train_base  >> fold_${fold}_"$(hostname)".log 2>&1
    done

done



