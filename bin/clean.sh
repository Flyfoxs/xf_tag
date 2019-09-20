#!/usr/bin/env bash
cd "$(dirname "$0")"
cd ..


rm ./cache/merge_feature*
rm ./cache/get_final_feature*
python spider/gen_file.py  > gen2.log 2>&1
cat ./input/zip/apptype_train.dat_p* > ./input/zip/apptype_train.dat
#./bin/test.sh





