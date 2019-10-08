#!/usr/bin/env bash

cd "$(dirname "$0")"
cd ..

./bin/deploy.sh $1

remote_host="aladdin1@$1"
remote_dir="~/felix/$(basename "$(pwd)")/*"


if [[ -z "$2" ]]; then
    rsync -avz --exclude-from './bin/exclude.txt' --max-size=1m  $remote_host:$remote_dir  ./
else
    rsync -avz --max-size=1m  $remote_host:$remote_dir  ./
fi

date

echo 'download from:' $remote_host:$remote_dir
