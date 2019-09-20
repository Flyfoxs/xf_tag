cd "$(dirname "$0")"

cd ..
best_arg="./imp/best_arg.h5"
if [ -f "$best_arg" ]; then
    echo "Already have best args in $best_arg"
else
    #生成当前最优参数,存放于目录 ./imp/
    echo "Try go take_snapshotf for best args, and save in $best_arg"
    python ./core/check.py  > snap_args.log 2>&1
fi

#提前对一些分析数据准备好本地缓存
python ./core/feature.py   > feature_prepare.log 2>&1

#根据生成的最优参数,预测缺少数据
python ./core/merge.py  --genfile > predict_block.log 2>&1

