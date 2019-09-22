
# 数据的准备
需要下载bert的预训练模型,然后在conf中修改对应的变量pretrained_path

下载地址: https://docs.google.com/uc?export=download&id=1W3WgPJWGVKlU9wpUYsdZuurAIFKvrl_Y


# 运行方式
   
*  OOF的生成: BERT
    
    以5折的方式运行程序,并且每一折都运行3次
    
    nohup ./bin/main.sh 3 &
    

*  融合OOF: 对生成的OOF合并生成提交文件:
    
    nohup python -u ./core/ensemble_new.py  main  >> ensemble_final.log 2>&1 &
    
*  生成的提交文件可见于 ./output/sub/
    

# 数据的爬取
* 百度
* 应用宝
* 豌豆荚
* 小米应用商店

# 数据清洗
- 异常数据的发现

    -  appname 为 #ffababab
        ffababab	    com.hisense.uifac
        FACTORY MENU	com.hisense.uifac
        工厂菜单	        com.hisense.uifac
    
    - appname 为 WXEntryActivity (138个)
        WXEntryActivity	    com.tencent.ft
        妖精的尾巴：魔导少年	com.tencent.ft
        
    - 百度手机助手(*)  
        百度手机助手(米聊)     
        百度手机助手(平安金管家)   
        百度手机助手(掌上电力)  



- 繁体字
    所有的繁体字转简体字


# 数据
- 数据增强(切割)
    由于bert等模型,都对输入的seq_len有字数限制. 余下的数据只能浪费.数据增强的一点是对浪费的数据进行利用

- 数据增强(不同源)
    由于第二期,是自行爬取数据,往往对于一条数据可以从多个源爬取数据.如果只选择一个源,则会对数据进行另外的一种浪费

- 数据增强(拼接)
    不同源的优先级
        由于查询存在模糊查询的情况,有package的查询或者name的查询. 并且不用应用商店的数据质量也不一样.
    
    不同长度的优先级
        每一个源头的数据,对于头部的文本,对准确性的判断权重更高. 可以对多个数据源的头部进行选取,拼接在一起形成一个新的实例
    
- 过拟合的避免
    数据增强后, 如果同一源头的数据如果切分到了训练集和验证集,会造成本地分数虚高.所以需要保证同源数据分到同一fold


# 算法
- 不同seq_len
    不同的seq_len 会有不同的表现,可以择优保留,然后融合

- 不同模型
    bert: 由于Bert有字数限制,最长512,并且太长效果并不是特别好.只能对局部信息进行利用
    lstm: LSTM对比Bert可以设置较大的训练窗口,利用大部分数据.和bert进行融合,是一个极大的补充.

- 克服抖动
    由于利用GPU进行训练,结果往往会有一定的抖动,可以对同一模型训练多轮,择优保留


# 后处理/融合
- 不同数据源
    数据增强,不仅仅针对训练集, 对测试集也做了对应的增强. 这样,同一条测试集,会有多个预测结果.这样对这些结果进行加权融合会有比较好的结果.
    

- 不同模型
    同一条测试集,会有多个模型进行预测,然后进行加权融合.

- 不同的切割序列
    不同模型,选择不同的seq_len,得到的不同结果,然后融合.