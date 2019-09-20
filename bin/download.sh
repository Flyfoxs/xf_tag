#!/usr/bin/env bash
cd "$(dirname "$0")"

cd ..

date
#rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/bk/df_jf/cache ./

#rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/xf_tag/output/spider ./output

rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/xf_tag/output/sub/*.* ./output/sub/

#rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/xf_tag/output/v75* ./
#
#rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/xf_tag/input/08*/ ./input/0823

#rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/xf_tag/input/zip/app*.dat ./input/zip/

#rsync -av  root@vm-ai-2:/apps/kdd_bd/output/*.* ./output/
#rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/kdd_bd/cache/get_fea*.* ./cache/
#rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/kdd_bd/output/sub/st_adj_0.68612_0.677874.csv ./output/sub/
#rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/xf_tag/input/jieba.* ./input
date