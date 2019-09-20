#!/usr/bin/env bash
cd "$(dirname "$0")"

cd ..


rsync -av ./score/blks/ hdpsbp@ai-prd-05:/users/hdpsbp/felix/df_jf/score/blks
rsync -av hdpsbp@ai-prd-05:/users/hdpsbp/felix/df_jf/score/blks ./score/blks/


rsync -av ./output/blocks/ hdpsbp@ai-prd-05:/users/hdpsbp/felix/df_jf/score/blks
rsync -av hdpsbp@ai-prd-05:/users/hdpsbp/felix/kdd_bd ./output/blocks/


rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/df_jf/imp  ./

date

