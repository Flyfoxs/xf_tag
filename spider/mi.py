
from multiprocessing import Process
import pandas as pd
from glob import glob
import re

import os
from time import time
from tqdm import tqdm
from file_cache.utils.util_log import timed, timed_bolck
from file_cache.cache import file_cache



def fix_name(ph2):
    ph2 = ph2.copy()
    #ph2['new_name'] = ph2['name']
    reg_list = ['应用市场（(.*)）', '百度手机助手\((.*)\)',
                '(搜狗浏览器)（(.*)）','(搜狗浏览器)\((.*)\)',
                '下载中心（(.*)）', '搜狗手机助手（(.*)）',
                '应用中心（(.*)）', '应用商店（(.*)）',
                '(.*)\-Android.*', '搜狗搜索\-(.*)',
                '(.*)\-锁屏精灵',
                ]
    for reg   in reg_list:
        ph2['name'] = ph2.name.apply(lambda val: re.sub(reg, r'\1', val))

    reg = '\(推荐\)|\(QQ浏览器\)|\(腾讯\)|\(首选\)|\(酷狗音乐\)|\(全民K歌\)|\(搜狗输入法\)|\(免费版\）|\(单机版\)|（中文版）|（推荐）'
    ph2['name'] = ph2.name.apply(lambda val: re.sub(reg, '', val).strip())

    convert_map = {
        '.*搜狗搜索加强版.*':'免费小说',
        #'.*双开助手.*':'双开助手',
        '#ffababab':'工厂菜单',

    }

    for k, v in convert_map.items():
        ph2['name'] = ph2.name.apply(lambda val: re.sub(k, v, val))

    return ph2

def fix_pkg(ph2):
    exclude_pkg = ['com.excelliance.dualaid', 'com.excean.dualaid',
                   'info.red.virtual',
                   #'sogou.mobile.explorer'
                   ]
    # def get_pkg_need_to_fix():
    #     final = get_final_feature()
    #     final.from_pkg = final.from_pkg.fillna(False)
    #     final['label'] = final.type_id.str.len() >= 1
    #     f_gp = final.groupby('pkg').label.agg({'cnt': 'count', 'train_test': 'nunique'})
    #     f_gp[['cnt_from_pkg', 'sum_from_pkg']] = final.groupby('pkg').from_pkg.agg(
    #         {'cnt_from_pkg': 'count', 'sum_from_pkg': 'sum'})
    #     f_gp = f_gp.loc[(f_gp.sum_from_pkg >= 2)].sort_values(['sum_from_pkg', 'cnt', 'train_test', ], ascending=False)
    #
    #     return f_gp
    def get_pkg_staic():
        df  = pd.read_csv('./input/0823/pkg_fix.csv')
        df.index = df.pkg
        return df.loc[~df.index.isin(exclude_pkg)]

    fix_pkg = get_pkg_staic().index.values
    ph2.pkg = ph2.pkg.apply(lambda val: f'{val},_fix' if val in fix_pkg else val)

    return ph2

def get_data_by_xm(pkg):
    try:
        from lxml import html
        import requests

        url = f'http://app.mi.com/details?id={pkg}'

        response = requests.get(url)
        tree = html.fromstring(response.content)

        desc = tree.xpath('/html/body/div[6]/div[1]/div[4]/p/text()')
        desc = ''.join(desc) if len(desc) > 0 else ''
        desc = desc.replace('\t', '')
        desc = desc.replace('\r', '')
        desc = desc.replace('\n', '').strip()

        cat_list = tree.xpath('/html/body/div[6]/div[1]/div[2]/div[1]/div/p[2]/text()[1]')

        app_name = tree.xpath('/html/body/div[6]/div[1]/div[2]/div[1]/div/h3/text()')[0]
        print(cat_list, app_name, desc)

        res = {
            'pkg': pkg,
            'name': app_name,
            'match_name': app_name,
            'cat_list': ','.join(cat_list),
            'desc': desc,
            'ct': pd.to_datetime('now'),
            'source': 'xm'
        }

    except Exception as e:
        print(e, html)
        res = {'pkg': pkg, 'source': 'xm', 'ct': pd.to_datetime('now'),}
    return res


def get_data_by_tx_name(name):
    try:
        from lxml import html
        import requests

        from requests.utils import quote

        name_new = quote(name)
        # url = f'https://android.myapp.com/myapp/search.htm?kw={name_new}'
        url = f'https://android.myapp.com/myapp/searchAjax.htm?kw={name_new}'
        response = requests.get(url)

        import json

        res = json.loads(response.text)

        app = res.get('obj').get('items')[0]
        pkg = app.get('pkgName')

        return get_data_by_tx_pkg(pkg, name)
        # res = {
        #     'pkg': pkg,
        #     'name': name,
        #     'match_name': app.get('appDetail').get('appName'),
        #     'cat_list': app.get('appDetail').get('categoryName'),
        #     'desc': app.get('appDetail').get('description'),
        #     'ct': pd.to_datetime('now'),
        #     'source': 'tx_name'
        # }

    except Exception as e:

        print(e, url)
        #raise e
        res = {'name': name, 'source': 'tx_name', 'ct': pd.to_datetime('now'), }
    return res


def get_data_by_tx_pkg(pkg, input_name=None):
    try:
        from lxml import html
        import requests

        url = f'https://android.myapp.com/myapp/detail.htm?apkName={pkg}'

        response = requests.get(url)
        tree = html.fromstring(response.content)

        desc = tree.xpath('//*[@id="J_DetAppDataInfo"]/div[1]/text()')
        desc = ''.join(desc) if len(desc) > 0 else ''
        desc = desc.replace('\t', '')
        desc = desc.replace('\r', '')
        desc = desc.replace('\n', '').strip()

        cat_list = tree.xpath('//*[@id="J_DetCate"]/text()')

        rate = tree.xpath('//*[@id="J_DetDataContainer"]/div//div[@class="com-blue-star-num"]/text()')[0]
        rate = rate or ''

        download = tree.xpath('//*[@id="J_DetDataContainer"]//div/div[@class="det-ins-num"]/text()')[0]
        download = download or ''

        app_name = tree.xpath('//*[@id="J_DetDataContainer"]/div/div[1]/div[2]/div[1]/div[1]/text()')[0]
        print(cat_list, app_name, desc)

        res = {
            'pkg': pkg +'_name' if input_name else pkg,
            'name': input_name or app_name,
            'match_name': app_name,
            'rate':float(rate.replace('分','')),
            'download':download,
            'cat_list': ','.join(cat_list),
            'desc': desc,
            'ct': pd.to_datetime('now'),
            'source': 'tx_pkg'
        }

    except Exception as e:
        print(e, html)
        res = {'pkg': pkg, 'source': 'tx_pkg', 'ct': pd.to_datetime('now'),}
    return res



def get_data_from_bd(name):
    try:
        from requests.utils import quote
        name_new = quote(name)
        url2 = f'http://www.baidu.com/s?wd={name_new}%20安卓版%20安卓'
        from lxml import html
        import requests
        #print(url2)
        response = requests.get(url2, timeout=5)
        #print(1)
        # print(response.text)
        tree = html.fromstring(response.content)
        # print(tree)
        # print(url2)
        # print(response.text)
        # desc = tree.xpath('/html/body/div[2]/div[2]/div[2]/div[1]/div[3]/div/div/text()[1]'
        # desc = tree.xpath('/html/body/div[2]/div[2]/div[2]/div[1]/div/div/div[1]/p/text()[1]')
        desc = tree.xpath('//*[@id="1" or @id="2"]//div[@class="c-abstract"]//text()')
        #print(2)
        desc = ''.join(desc)
        import re
        desc = re.sub('[0-9]*年[0-9]*月[0-9]*日.-.', '', desc)
        #print(3)
        res = {
            'name': name,
            #'match_name': name,
            'desc_bd': desc,
            'ct': pd.to_datetime('now'),

        }
    except Exception as e:
        print(name, url2)
        print(e)
        res = {'name': name, 'ct': pd.to_datetime('now'), }
    print(4)
    return res



def get_data_from_wdj(name):
    try:
        #print('=' * 20)
        from lxml import html
        import requests
        from requests.utils import quote
        url, url2 = None,  None
        name_new = quote(name)
        url = f'https://www.wandoujia.com/search?key={name_new}&source=index'
        url2 = None
        response = requests.get(url)
        tree = html.fromstring(response.content)
        link = tree.xpath('//*[@id="j-search-list"]/li[2]/a')[0]
        url2 = link.get("href")
        #print(url2)
        response = requests.get(url2)
        tree = html.fromstring(response.content)

        desc = tree.xpath('/html/body/div[2]/div[2]/div/div/*/div[@itemprop="description"]//text()')
        desc = ''.join(desc) if len(desc) > 0 else ''
        desc = desc.replace('\t', '')
        desc = desc.replace('\r', '')
        desc = desc.replace('\n', '').strip()

        dp = tree.xpath('/html/body/div[2]/div[2]/div[2]//div[@class="editorComment"]/div[@class="con"]/text()')
        dp = dp[0] if len(dp) > 0 else ''
        dp = dp.strip()

        closed_ids = tree.xpath('/html/body/div[2]/div[2]/div[2]/div[2]/div[3]/ol/li[*]/a[1]')  # [0]

        match_name = tree.xpath('/html/body/div[2]/div[2]/div[1]//div/p/span[@itemprop="name"]/text()')[0]

        cat_list = tree.xpath('/html/body/div[2]/div[2]/div[2]/div[2]/div[1]/dl/dd[2]/a/text()')
        tag_list = tree.xpath('/html/body/div[2]/div[2]/div[2]/div[2]/div[1]/dl/dd[3]/div/div/a/text()')
        closed_ids = [item.get('href').split('/')[-1] for item in closed_ids]
        path = tree.xpath('/html/body/div[2]/div[1]/div[2]//span[@itemprop="title"]/text()')[0]

        wd_id = url2.split('/')[-1]
        res = {
            'wd_id': wd_id,
            'name': name,
            'match_name': str(match_name),
            'tag_list': ','.join(tag_list),
            'cat_list': ','.join(cat_list),
            'closed_ids': ','.join(closed_ids),
            'path':     str(path),
            'desc':     desc,
            'dp': dp.strip().replace('\t', ''),  # 点评
            'ct':pd.to_datetime('now'),
            'source':'wdj'

        }

        #print(res)

    except Exception as e:
        print(name)
        print(name, url2 or url)
        print(e)
        res = {'name': name,'ct':pd.to_datetime('now'), }

        # if url2 is None:
        #     res = get_data_from_bd(name)

    return res


def get_data_from_bdsj(name):
    try:
        from requests.utils import quote
        name_new = quote(name)
        url = f'https://shouji.baidu.com/s?wd={name_new}&data_type=app&f=header_all%40input%40btn_search'
        from lxml import html
        import requests
        print(url)
        url2=None
        response = requests.get(url)
        #print(response.content)
        tree = html.fromstring(response.content)
        link = tree.xpath('//a[@class="app-name"]')[0]
        print('===',link)
        url2 = f'https://shouji.baidu.com{link.get("href")}'
        print('url2==',url2)
        response = requests.get(url2)
        tree = html.fromstring(response.content)

        #desc = tree.xpath('//*[@id="doc"]//p[@class="content content_hover"]/text()')
        desc = tree.xpath('//*[@id="doc"]//p[@class="content content_hover" or @class="content"]/text()')
        desc = ''.join(desc) if len(desc) > 0 else ''
        desc = desc.replace('\t', '')
        desc = desc.replace('\r', '')
        desc = desc.replace('\n', '').strip()
        print(desc)
        dp = tree.xpath('//*[@id="doc"]//div/span[@class="head-content"]/text()')
        dp = dp[0] if len(dp) > 0 else ''
        dp = dp.strip()
        print('dp===',dp)

        #closed_ids = tree.xpath('/html/body/div[2]/div[2]/div[2]/div[2]/div[3]/ol/li[*]/a[1]')  # [0]

        match_name = tree.xpath('//*[@id="doc"]/div[2]/div/div[1]/div/div[2]/h1/span/text()')[0]
        print('match_name', match_name,  'name', name)
        cat_list = tree.xpath('//*[@id="doc"]/div[1]/div//span/a/text()')[1:]
        print('cat_list', cat_list)

        res = {
            #'wd_id': wd_id,
            'name': name,
            'match_name': str(match_name),
            #'tag_list': ','.join(tag_list),
            'cat_list': ','.join(cat_list),
            #'closed_ids': ','.join(closed_ids),
            #'path':     str(path),
            'desc':     desc,
            'dp': dp.strip().replace('\t', ''),  # 点评
            'ct':pd.to_datetime('now'),
            'source':'bdsj'

        }




    except Exception as e:
        print(name)
        print(name, url2 or url)
        print(e)
        res = {'name': name,'ct':pd.to_datetime('now'), }

    #print(res)
    return res


def get_todo_list_pkg(source):
    #wdj = merge_file('./output/spider/wdj', replace=False)

    good_df = merge_file(f'./output/spider/{source}', replace=False)

    if good_df is None or len(good_df)==0:
        good_pkg =[]
    else:
        good_pkg = good_df.pkg

    print(f'Good pkg:{len(good_pkg)}')


    ph2 = get_ph2()[['name', 'pkg']]

    pkg = ph2.loc[~ph2.pkg.isin(good_pkg)].pkg.drop_duplicates()
    print(f'TODO list:{len(pkg)}')
    return pkg.to_list()


def get_ph2(fix=True):
    #df1 = pd.read_csv('./input/0823/final_apptype_train.dat', names=['name', 'pkg', 'type_id'], sep='\t')
    df1 = pd.read_csv('./input/0823/final_apptype_train.dat')

    # df1_index = get_train_ph2_index()
    # df1['id']=df1_index['id']

    df2 = pd.read_csv('./input/0823/appname_package.dat', names=['id', 'name', 'pkg'], sep='\t')

    ph2 = pd.concat([df1, df2])
    if fix!=True:
        return ph2
    else:
        ph2.name = ph2.name.astype(str).str.lower()
        import re
        #ph2.name = ph2.name(lambda val: re.sub('(推荐)|(QQ浏览器)', '', val))

        ph2.pkg = ph2.pkg.fillna('').astype(str).str.lower()

        ph2 = ph2.fillna('')
        #Fix same package have diff name
        ph2 = pd.merge(ph2, get_pkg_name(), how='left', on='pkg')
        ph2.new_name = ph2.new_name.fillna('').astype(str)
        ph2['name'] = ph2['name'].astype(str).str.lower()

        # ph2 = ph2.fillna('')
        #
        # ph2['name'] = ph2.apply(lambda row: ','.join([row.pkg]) if row['name'].lower() == 'WXEntryActivity'.lower() else row['name'],
        #                         axis=1)  # .astype(str)
        # ph2['name'] = ph2['name'].astype(str)
        #

        ph2 = fix_name(ph2)
        ph2 = fix_pkg(ph2)
        ph2['name'] = ph2.apply(lambda row: row['new_name'] if len(row['new_name'])>0 else row['name'], axis=1)

        return ph2.reset_index(drop=True)

def get_pkg_name():
    df = pd.read_csv('./input/0823/pkg_name.csv')
    df = df.drop_duplicates('pkg')
    return df
    # ph2 = get_ph2(fix=False)
    # todo = ph2.loc[ph2.name == 'WXEntryActivity']
    # print(todo.shape)
    #
    # new = ph2.loc[ph2.pkg.isin(todo.pkg) & (ph2.name != 'WXEntryActivity')]
    #
    # final = get_final_feature()
    # new = pd.merge(new, final[['id', 'from']], how='left', on='id')
    # new['from'] = new['from'].replace({'bd': 'wwww'})
    #
    # new_single = new.sort_values(['pkg', 'from', 'name', ], ascending=[False, True, False]).drop_duplicates(
    #     'pkg')  # .shape
    #
    # df = pd.merge(new, new_single[['pkg', 'name']], how='left', on='pkg').sort_values('name_y')
    # df['new_name'] = df.name_y
    # df[['pkg', 'new_name']].to_csv('./input/0823/pkg_name.csv', index=None, header=True)


def get_todo_list_tx_name(source):
    ph2 = get_ph2()

    from glob import glob
    tx_pkg = pd.concat([pd.read_hdf(file, 'wdj') for file in glob(f'./output/spider/tx_pkg/*.h5')])
    tx_pkg = tx_pkg.dropna()
    tx_pkg = tx_pkg.loc[tx_pkg.desc.str.len() >= 20]


    ph2 = ph2.loc[~ph2.pkg.isin(tx_pkg.pkg)]
    # ph2 = ph2.loc[~ph2.name.isin(tx_pkg.name)]

    exist_df = merge_file(f'./output/spider/{source}', replace=False)

    if not exist_df.empty:
        ph2 = ph2.loc[~ph2.name.isin(exist_df.name)]

    return ph2.name.drop_duplicates().sort_values().to_list()


def get_todo_list_bd():
    ph2 = get_ph2()

    todo = ph2.name.drop_duplicates()

    print(f'The todo list get from wdj is:{len(todo)}')
    if len(glob('./output/spider/bd/*.h5'))>0:
        good_df = pd.concat([pd.read_hdf(file, 'wdj') for file in glob('./output/spider/bd/*.h5')])

        good_df  = good_df.fillna('')
        good_df  = good_df.loc[good_df.desc_bd.str.len()>10]
        #good_df = good_df.loc[good_df.desc_bd.str.len()>=threshold]

        return sorted(list(set(todo) - set(good_df.name.astype(str))))
    else:
        return sorted(list(set(todo)))


def get_todo_list_name(source):
    final = get_final_feature()
    final['from'].value_counts()

    sj = merge_file('./output/spider/bdsj')
    sj = sj.loc[sj.apply(lambda row: row['name'] == row['match_name'], axis=1)]  # .shape
    print(sj.shape)

    final.loc[final['from'] != 'all_m'].head()
    # 5246
    print(final.columns)
    (sj.desc.str.len() // 20).value_counts()
    return sj.loc[(sj.desc.str.len() == 0).values].name.to_list()


def get_todo_list_wdj(threshold=20):

    res_list = []

    df = get_ph2()

    from glob import glob
    good_df = pd.concat([pd.read_hdf(file, 'wdj') for file in glob('./output/spider/wdj/*.h5')])
    good_df.ct = good_df.ct.astype(str)
    if len(good_df)>0:
        good_df = good_df.sort_values('ct').drop_duplicates('name', keep='last')
        good_df.fillna('')
        good_df.desc = good_df.desc.str.strip()
        good_df = good_df.loc[good_df.desc.str.len()>=threshold]
        exist_list = good_df.apply(lambda row: row.to_dict(),  axis=1)
        print(f'Already get {len(exist_list)} rows')
        res_list.extend(exist_list)
        name_exist = good_df.name.to_list()
        name_list = df.loc[~df.name.isin(name_exist)].name.sort_values().astype(str).to_list()
    else:
        name_list = df.name.sort_values().drop_duplicates().astype(str).to_list()
    print(len(name_list))
    return  list(set(name_list))


def process_name_list(name_list, source):
    import threading
    #thread_name =  threading.currentThread().getName()
    pid = os.getpid()
    local_list = []
    print(f'\nThere are {len(name_list)} need to process for this batch#{pid}\n')
    file = f'./output/spider/{source}/{source}_{time()}.h5'
    for sn, name in enumerate(tqdm(name_list, desc=f'{source}, Process#{pid}')):
        if source == 'wdj':
            res = get_data_from_wdj(name)
        elif source == 'bd':
            res = get_data_from_bd(name)
        elif source == 'xm':
            res = get_data_by_xm(name)
        elif source == 'tx_pkg':
            res = get_data_by_tx_pkg(name)
        elif source == 'tx_name':
            res = get_data_by_tx_name(name)
        # elif source == '360':
        #     res = get_data_from_360(name)
        elif source == 'bdsj':
            res = get_data_from_bdsj(name)
        local_list.append(res)
        if sn % 100 == 99:
            print(f'{len(local_list)} rows save to file,  sn:{sn}')
            pd.DataFrame(local_list).to_hdf(file, 'wdj', mode='w')

    pd.DataFrame(local_list).to_hdf(file, 'wdj',mode='w')

    print(f'{len(local_list)} res save to file:{file}')

#
# def extend_df():
#     tqdm.pandas(desc="my bar!")
#     good_df = get_final()
#     good_df.desc = good_df.desc.str.strip()
#     todo = good_df.loc[(
#                         (good_df.desc.str.len() <= 20) |
#                         pd.isna(good_df.desc))
#     ]
#     print(len(todo))
#     desc_bd = todo.name.progress_apply(lambda name: get_data_from_bd(name))
#     good_df['desc_bd'] = desc_bd
#
#     good_df.to_hdf('./input/final.h5', 'desc')

# def get_final():
#     if os.path.exists('./input/final.h5'):
#         return  pd.read_hdf('./input/final.h5','desc')
#     else:
#         good_df = pd.concat([pd.read_hdf(file, 'wdj') for file in glob('./output/spider/*.h5')])
#         good_df.ct = good_df.ct.astype(str)
#         good_df = good_df.sort_values('ct').drop_duplicates('name', keep='last')
#         good_df = good_df.reset_index(drop=True)
#         return good_df


def spider_name_list(name_list, source='wdj'):
    name_list_len = len(name_list)
    if source == '360':
        thread_num = 2
    else:
        thread_num = 10

    step = name_list_len//thread_num
    for partition_sn in range(thread_num):
        begin , end = step*partition_sn, step*(partition_sn+1)
        if partition_sn ==thread_num-1:
            end =  name_list_len
        print(f'\nthread:{partition_sn}, begin:{begin},end:{end}, total:{name_list_len}')
        p = Process(target=process_name_list, args=(name_list[begin:end],source), name=f'p{partition_sn}')
        p.start()


#
# @file_cache()
# def get_final():
#     wdj = pd.concat([pd.read_hdf(file, 'wdj') for file in glob('./output/spider/wdj/*.h5')])
#     wdj.ct = wdj.ct.astype('str')
#     wdj = wdj.sort_values('ct').drop_duplicates('name', keep='last')
#     bd = pd.concat([pd.read_hdf(file, 'wdj') for file in glob('./output/spider/bd/*.h5')])
#     bd = bd.sort_values('ct').drop_duplicates('name', keep='last')
#
#     return pd.merge(wdj, bd, how='left', on='name')


@timed()
def merge_file(fold, replace=False):
    try:
        df = pd.concat([pd.read_hdf(file, 'wdj') for file in glob(f'{fold}/*.h5')])
        df.ct = df.ct.astype('str')
        df['name'] = df['name'].fillna('').astype('str').str.lower()

        check_col = ['name']
        if 'pkg' in df.columns:
            check_col.append('pkg')

        if 'desc_bd' in df.columns:
            df = df.rename({'desc_bd':'desc'}, axis=1)

        df = df.sort_values('ct').drop_duplicates(check_col, keep='last')

        if 'match_name' not in df.columns:
            df['match_name'] = df['name']

        df['match_name'] = df['match_name'].astype(str).str.lower()

        df['desc'] = df['desc'].fillna('').astype(str)
        df['desc'] = df.desc.apply(lambda val: '' if val.startswith('??') else val)

        if replace:
            for file in glob(f'{fold}/*.h5'):
                os.rename(file, f'{file}_del')

            file = f'{fold}/merge_{len(df)}.h5'
            df.to_hdf(file, 'wdj')

            print(f'merge file in  fold to :{file}')
            return df
        else:
            return df
    except Exception as e:
        print(e)
        return pd.DataFrame()


def expend_to_test(df, base='match_name'):
    df = df.copy()
    df = df.drop(['rate','len_'], axis=1, errors='ignore')
    if base=='match_name':
        del df['name']
        df.match_name  = df.match_name.astype(str)

    ph2 = get_ph2()

    df_name = df.copy().drop_duplicates(base)
    df_name = df_name.drop('pkg', axis=1, errors='ignore')
    df_name = df_name.rename({'desc_ex': 'desc_name'}, axis=1)
    ph2 = pd.merge(ph2, df_name, how='left', left_on='name', right_on=base)

    if 'pkg' in df.columns:
        df_pkg = df.copy().drop_duplicates('pkg')
        df_pkg = df_pkg.rename({'desc_ex': 'desc_pkg', 'match_name':'match_name_pkg'}, axis=1)
        df_pkg = df_pkg.drop(['name'], axis=1, errors='ignore')

        print(ph2.columns, df_pkg.columns, ph2.shape, df_pkg.shape)
        ph2 = pd.merge(ph2, df_pkg, how='left', on='pkg')

        ph2['match_name'] = ph2.apply(lambda row: row.match_name_pkg if pd.notna(row.match_name_pkg) else row.match_name, axis=1)

        ph2['from_pkg'] = ph2.desc_pkg.apply(lambda val: pd.notna(val) and len(str(val)) >=1 )

        ph2.desc_pkg = ph2.desc_pkg.fillna('').astype(str)
        ph2.desc_name = ph2.desc_name.fillna('').astype(str)

        def merge_desc(row):
            if len(row.desc_pkg) > 128:
                return row.desc_pkg
            elif len(row.desc_name) > 128:
                return row.desc_name
            elif len(row.desc_pkg) == 0:
                return row.desc_name
            elif len(row.desc_name) == 0 :
                return row.desc_pkg
            else:
                return row.desc_pkg + '/' +row.desc_name

        ph2['desc_name'] = ph2.apply( lambda row:  merge_desc(row) , axis=1)

        #del ph2['desc_pkg']

    return ph2

#
# def expend_to_test(df):
#     ph2 = get_ph2()
#
#     df_name = df.copy().drop_duplicates('name')
#     df_name = df_name.drop('pkg', axis=1, errors='ignore')
#     df_name = df_name.rename({'desc_ex': 'desc_name'}, axis=1)
#     ph2 = pd.merge(ph2, df_name, how='left', on='name')
#
#     if 'pkg' in df.columns:
#         df_pkg = df.copy().drop_duplicates('pkg')
#         df_pkg = df_pkg.rename({'desc_ex': 'desc_pkg'}, axis=1)
#         del df_pkg['name']
#         print(ph2.columns, df_pkg.columns)
#         ph2 = pd.merge(ph2, df_pkg, how='left', on='pkg')
#
#         ph2['desc_name'] = ph2.apply(
#             lambda row: row.desc_pkg if pd.notna(row.desc_pkg) and len(str(row.desc_pkg)) > 3 else row.desc_name,
#             axis=1)
#
#     return ph2

@timed()
@file_cache()
def merge_feature(source, base='match_name'):
    col_order = {
        'bd':     [            'desc'],
        'wdj':    ['cat_list', 'desc', 'dp', ],
        'bdsj':   ['cat_list', 'desc', 'dp'  ],
        'gg':     ['cat_list', 'desc'],
        'xm':     ['cat_list', 'desc'],
        'tx':     ['cat_list', 'desc'],
        'tx_pkg': ['cat_list', 'desc'],
        'tx_name':['cat_list', 'desc'],
        'all': ['cat_list', 'desc'],
    }
    df_dict = {}
    # for source, col_list in col_order.items():
    col_list = col_order.get(source)
    old_list = col_list.copy()
    col_list = col_list + ['name','match_name']
    if source == 'bd':
        print(source, col_list)
        df = merge_file('./output/spider/bd', replace=False).loc[:, col_list]
    elif source == 'xm':
        col_list.append('pkg')
        df = merge_file('./output/spider/xm', replace=False).loc[:, col_list]
        # df = merge_file('./output/spider/bd', replace=False).loc[:, col_list]
    elif source == 'tx_pkg':
        col_list.append('pkg')
        df = merge_file('./output/spider/tx_pkg', replace=False).loc[:, col_list]
    elif source == 'tx_name':
        df = merge_file('./output/spider/tx_name', replace=False).loc[:, col_list]
    elif source == 'tx':
        col_list.append('pkg')
        col_list.append('rate')
        #df_xm = merge_file('./output/spider/xm', replace=False).loc[:, col_list]
        df_pkg = merge_file('./output/spider/tx_pkg', replace=False).loc[:, col_list]
        df_name = merge_file('./output/spider/tx_name', replace=False).loc[:, col_list]
        # print(col_list)
        df = pd.concat([df_pkg, df_name], axis=0).reset_index(drop=True)
        df['len_'] = df.desc.str.len()
        df = df.loc[df.len_>0]
        df = df.sort_values([base,'len_', 'rate'], ascending=False)
    elif source == 'wdj':
        df = merge_file('./output/spider/wdj', replace=False).loc[:, col_list]
    elif source == 'bdsj':
        df = merge_file('./output/spider/bdsj', replace=False).loc[:, col_list]

    elif source == 'gg':
        df = merge_file('./output/spider/gg', replace=False).loc[:, col_list]
    else:# source == 'all':
        col_list.append('pkg')
        df_xm = merge_file('./output/spider/xm', replace=False).loc[:, col_list]

        col_list.append('rate')
        df_pkg = merge_file('./output/spider/tx_pkg', replace=False).loc[:, col_list]
        df_name = merge_file('./output/spider/tx_name', replace=False).loc[:, col_list]
        #df_wdj = merge_file('./output/spider/wdj', replace=False).loc[:, col_list]
        # print(col_list)
        df = pd.concat([df_pkg, df_xm, df_name], axis=0).reset_index(drop=True)
        print(df.columns)
        df['len_'] = df.desc.str.len()
        df = df.loc[df.len_ > 0]
        df = df.sort_values([base,'len_', 'rate'], ascending=False) #饥饿鲨：进化

    def join_row(row):
        try:
            return ','.join(row.dropna().values.astype(str))
        except Exception as e:
            print(e, row)
            return 'ERROR'

    df['desc_ex'] = df.loc[:, old_list].apply(lambda row: join_row(row), axis=1)

    #df['desc_ex'] = df['desc_ex'].apply(lambda val: re.sub('default,|更新内容：|default|\r|\n', '', val))

    df = df.drop(old_list, axis=1)

    return expend_to_test(df, base)



def clean_feature(df):
    print(df.columns)
    df = df.loc[df.desc_name.str.len() >= 2]
    return df





def filter_desc(desc):
    desc = str(desc)
    reg = re.compile(r"""
        (游戏语言:.*?\s)
        |(游戏类型:.*?\s)
        #|(游戏类别:.*?\s)
        |(游戏授权:.*?\s)
        |(版本游戏大小:.*?\s)

        |(软件语言:.*?\s)
        #|(软件类型:.*?\s)
        #|(软件类别:.*?\s)
        |(软件授权:.*?\s)
        |(软件大小:.*?\s)
        |(软件厂商:.*?\s)

        |(应用授权:.*?\s)
        |(应用类型:.*?\s)
        |(应用语言:.*?\s)
        |(应用大小:.*?\s)

        |(时间:.{10}?\s((.{5}\s))?) 

    """, re.VERBOSE)

    desc = reg.sub(' ', desc)

    reg = re.compile(r"""
                        \xa0
                        |default,
                        |更新内容：
                        |default
                        |扫描二维码下载应用至手机
                        |\r|\n
                        |◆{2,}|■{2,}
                        |(时间:.{10}?\s((.{5}\s))?) 
                        |(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?
                        |\s{2,}
                        |\.{2,}
                        |,{2,}
                        |-{2,}
                        |—{2,}
                        |={2,}
                        |！{2,}
                        |!{2,}
                        |\*{2,}
                        |,\_fix
                    """, re.VERBOSE)

    desc = reg.sub(',', desc)
    desc = reg.sub(',', desc)
    return desc

@timed()
@file_cache()
def get_final_feature():
    """
    match_name, name, pkg, pure_name
    """

    #max_len = int(max_len)

    feature_list = []

    tmp = merge_feature('all', 'match_name')
    feature_list.append(tmp)

    tmp = merge_feature('wdj', 'match_name')
    feature_list.append(clean_feature(tmp)[['desc_name']].add_suffix('_mwdj'))

    tmp = merge_feature('bdsj', 'match_name')
    feature_list.append(clean_feature(tmp)[['desc_name']].add_suffix('_msj'))

    tmp = merge_feature('gg', 'match_name')
    feature_list.append(clean_feature(tmp)[['desc_name']].add_suffix('_mgg'))

    tmp = merge_feature('bd', 'match_name')
    # tmp = tmp.rename({'desc_name':'desc_bd'}, axis=1)
    feature_list.append(clean_feature(tmp)[['desc_name']].add_suffix('_bd'))

    tmp = merge_feature('tx', 'name')
    # tmp = tmp.rename({'desc_name':'desc_tx'}, axis=1)
    feature_list.append(clean_feature(tmp)[['desc_name']].add_suffix('_tx'))

    tmp = merge_feature('wdj', 'name')
    # tmp = tmp.rename({'desc_name':'desc_wdj'}, axis=1)
    feature_list.append(clean_feature(tmp)[['desc_name']].add_suffix('_wdj'))

    with timed_bolck(f'concat#{len(feature_list)} df'):
        df = pd.concat(feature_list, axis=1)
        print(df.columns)

    # df = pd.concat(feature_list).drop_duplicates('id')

    def merge_row(row):
        """
        todo: check name in or not
        todo: check: add name to every section
        :param row:
        :return:
        """
        col_list = ['name', 'desc_name',
                    'desc_name_mwdj', 'desc_name_msj',
                    'desc_name_bd', 'desc_name_mgg', 'desc_name_tx', 'desc_name_wdj',
                    'pkg']
        col_from = ','.join(row.loc[col_list].dropna().index)
        col_from = col_from.replace('desc_name,', 'mtx,')
        col_from = col_from.replace('desc_name_', '')
        col_from = col_from.replace('name,', '')

        #动态截取描述长度
        #for i in range(1,500):
        desc_list = [item for item in row.loc[col_list].dropna()]
        desc = ','.join(desc_list)
            # if len(desc)>=700:
            #     break

        if str(row.pkg).startswith('dkplugin.'):
            desc = '多开,工具,dkplugin,' + desc

        if str(row['name']) not in desc[:128]:
            # print(f'Name not in desc for {row["name"]},  desc:{desc[:128]}')
            desc = str(row['name']) + ',' + desc

        return pd.Series({'desc_name': filter_desc(desc), 'col_from': col_from})

    with timed_bolck(f'Merge col'):
        df[['desc_name', 'col_from']] = df.apply(lambda row: merge_row(row), axis=1)

    from core.feature import get_app_type
    type_name = get_app_type()
    type_name = type_name.set_index('type_id')
    type_name.index = type_name.index.astype(str)
    df['type_name'] = df.type_id.astype(str).replace(type_name.to_dict()['type_name'])
    out_list = ['id', 'pkg', 'name', 'match_name', 'desc_name', 'type_id', 'col_from']
    return df[[col for col in out_list if col in df.columns]]

def get_train_ph2_index():
    # from spider.mi import *
    # vs = get_final_feature()
    # vs = vs.loc[vs.type_id.str.len() == 0]
    # vs = vs.set_index('id')

    df1_index = pd.read_csv('./input/0823/index.dat')
    return df1_index

if __name__ == '__main__':
    #get_data_from_bd('安卓版')
    import sys
    if len(sys.argv)<=1:
        source = 'wdj'
    else:
        source = str(sys.argv[1])

    print('source',source)
    if source == 'wdj':
        name_list = get_todo_list_wdj(30)
    elif source == 'bd':
        name_list = get_todo_list_bd()
    elif source == 'xm':
        name_list = get_todo_list_pkg(source)
    elif source== 'tx_pkg':
        name_list = get_todo_list_pkg(source)
    elif source== 'tx_name':
        name_list = get_todo_list_tx_name(source)
    elif str(source) in ['360', 'bdsj']:
        name_list = get_todo_list_name(source)

    print(f'Todo List is {len(name_list)} for source:{source}')
    spider_name_list(name_list, source)

    #merge_file(f'./output/spider/{source}', True)


""""
http://so.cr173.com/search/d/%E5%BE%AE%E4%BF%A1_all_rank.html
http://www.baidu.com/s?wd=埃达之光%20安卓

nohup python -u  spider/mi.py bd  > bd.log 2>&1&
nohup python -u  spider/mi.py wdj  > wdj.log 2>&1&
nohup python -u  spider/mi.py xm  >  xm.log 2>&1&
nohup python -u  spider/mi.py 360  >  360.log 2>&1&

nohup python -u  spider/mi.py tx_pkg  >  tx_pkg.log 2>&1&
nohup python -u  spider/mi.py tx_name  >  tx_name.log 2>&1&

nohup python -u  spider/mi.py bdsj  >  bdsj.log 2>&1&
"""