#!/bin/bash
cd "$(dirname "$0")"
cd ..

version=v3
max_bin=1
cut_ratio=0.15
min_len_ratio=0.8
echo ./cache/${version}*.*
rm -rf ./cache/${version}*.*
mv ./output/stacking/${version}*.* ./output/bk_stacking/
for fold in {0..4};
do
    python -u ./core/bert.py  --fold=${fold} --version=${version}  --max_bin=${max_bin} --cut_ratio=${cut_ratio} --min_len_ratio=${min_len_ratio} train_base  >> ${version}_fold_${fold}_"$(hostname)".log 2>&1
done

version=v4
max_bin=1
cut_ratio=0.2
min_len_ratio=0.8
echo ./cache/${version}*.*
rm -rf ./cache/${version}*.*
mv ./output/stacking/${version}*.* ./output/bk_stacking/
for fold in {0..4};
do
    python -u ./core/bert.py  --fold=${fold} --version=${version}  --max_bin=${max_bin} --cut_ratio=${cut_ratio} --min_len_ratio=${min_len_ratio} train_base  >> ${version}_fold_${fold}_"$(hostname)".log 2>&1
done


version=v5
max_bin=2
cut_ratio=0.1
min_len_ratio=0.8
echo ./cache/${version}*.*
rm -rf ./cache/${version}*.*
mv ./output/stacking/${version}*.* ./output/bk_stacking/
for fold in {0..4};
do
    python -u ./core/bert.py  --fold=${fold} --version=${version}  --max_bin=${max_bin} --cut_ratio=${cut_ratio} --min_len_ratio=${min_len_ratio} train_base  >> ${version}_fold_${fold}_"$(hostname)".log 2>&1
done

