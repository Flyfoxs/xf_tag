#!/usr/bin/env bash
cd "$(dirname "$0")"

remote_host="hdpsbp@ai-prd-07"
remote_dir="/users/hdpsbp/felix/"

cd ..

if [[ -z "$1" ]]; then
    rsync -avz --exclude-from './bin/exclude.txt' $(pwd) $remote_host:$remote_dir
else
    rsync -avz $(pwd) $remote_host:$remote_dir
fi

date

echo $remote_host:$remote_dir

#rsync -av  ./output/0.70180553000.csv hdpsbp@ai-prd-07:/users/hdpsbp/felix/df_jf/output/


#rsync -av hdpsbp@ai-prd-04:/users/hdpsbp/felix/kdd_bd   /apps/


#rsync -av  ./input/tmp hdpsbp@ai-prd-07:/users/hdpsbp/felix/kdd_bd/input

#rsync -av   ./output/sub/?.csv hdpsbp@ai-prd-07:/users/hdpsbp/felix/kdd_bd/output/sub

