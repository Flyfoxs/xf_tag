如果此文件夹是空,需要运行下列命令生成input文件

# 爬取数据
nohup python -u  spider/mi.py bd  > bd.log 2>&1&
nohup python -u  spider/mi.py wdj  > wdj.log 2>&1&
nohup python -u  spider/mi.py xm  >  xm.log 2>&1&
nohup python -u  spider/mi.py 360  >  360.log 2>&1&

nohup python -u  spider/mi.py tx_pkg  >  tx_pkg.log 2>&1&
nohup python -u  spider/mi.py tx_name  >  tx_name.log 2>&1&

nohup python -u  spider/mi.py bdsj  >  bdsj.log 2>&1&


# 生成input数据
./bin/clean.sh
