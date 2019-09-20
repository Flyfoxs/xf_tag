cd "$(dirname "$0")"

cd ..

#rm -rf cache/get_feature_target*compare*.h5
nohup python -u code_felix/core/compare.py >> compare_"$(hostname)".log 2>&1 &
