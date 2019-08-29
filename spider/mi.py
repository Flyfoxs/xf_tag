def get_data_by_pkg(pkg):
    try:
        from bs4 import BeautifulSoup
        import requests
        # html =  f'https://play.google.com/store/apps/details?id={pkg}'

        html = f'http://app.mi.com/details?id={pkg}'

        # 获取字符串格式的html_doc。由于content为bytes类型，故需要decode()
        html_doc = requests.get(html).content.decode()
        # 使用BeautifulSoup模块对页面文件进行解析
        soup = BeautifulSoup(html_doc, 'html.parser')

        des = soup.select('body > div.main > div.container.cf > div.app-text > p')

        tag = soup.select('body > div.main > div.container.cf > div.app-intro.cf > '
                          'div.app-info > div > p.special-font.action')

        app_name = soup.select('body > div.main > div.container.cf > div.app-intro.cf > div.app-info > div > h3')

        desc = des[0].contents[0]

        app_name = app_name[0].contents[0]

        app_type = tag[0].text.split('|')[0].split('：')[1]

        print(app_name, app_type, desc)
        return app_name, app_type, desc
    except Exception as e:
        print(html)
        return None, None, None


def get_data_from_wdj(name):
    try:
        #print('=' * 20)
        from lxml import html
        import requests
        from requests.utils import quote
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

        desc = tree.xpath('/html/body/div[2]/div[2]/div[2]/div/*/div[@itemprop="description"]//text()')
        desc = ''.join(desc) if len(desc) > 0 else 'No desc'
        desc = desc.replace('\t', '')
        desc = desc.replace('\r', '')
        desc = desc.replace('\n', '')

        dp = tree.xpath('/html/body/div[2]/div[2]/div[2]//div[@class="editorComment"]/div[@class="con"]/text()')
        dp = dp[0] if len(dp) > 0 else 'No desc'

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
            'match_name': match_name,
            'tag_list': ','.join(tag_list),
            'cat_list': ','.join(cat_list),
            'closed_ids': ','.join(closed_ids),
            'path':path,
            'desc': desc,
            'dp': dp.strip().replace('\t', ''),  # 点评
            'ct':pd.to_datetime('now'),
            'source':'wdj'

        }

        #print(res)

    except Exception as e:
        print(name, url2 or url)
        print(e)
        res = {'name': name,'ct':pd.to_datetime('now'), }
    return res


import pandas as pd

df1 = pd.read_csv('./input/0823/final_apptype_train.dat', names=['name', 'pkg', 'type_id'], sep='\t')
df2 = pd.read_csv('./input/0823/appname_package.dat', names=['id', 'name', 'pkg'], sep='\t')

df = pd.concat([df1, df2])

from time import time
from tqdm import tqdm


res_list = []

import os
from glob import glob
good_df = pd.concat([pd.read_hdf(file, 'wdj') for file in glob('./output/spider/*.h5')])

if len(good_df)>0:
    good_df = good_df.drop_duplicates('name')
    #good_df = good_df.loc[~((good_df.desc=='No desc') | pd.isna(good_df.desc))]
    exist_list = good_df.apply(lambda row: row.to_dict(),  axis=1)
    print(f'Already get {len(exist_list)} rows')
    res_list.extend(exist_list)
    name_exist = good_df.name.to_list()
    name_list = df.loc[~df.name.isin(name_exist)].name.sort_values().drop_duplicates().to_list()
else:
    name_list = df.name.sort_values().drop_duplicates().to_list()
print(len(name_list))


def process_name_list(name_list):
    import threading
    #thread_name =  threading.currentThread().getName()
    pid = os.getpid()
    local_list = res_list.copy()
    print(f'\nThere are {len(name_list)} need to process for this batch#{pid}\n')
    file = f'./output/spider/wdj_{time()}.h5'
    for sn, name in enumerate(tqdm(name_list, desc=f'Process#{pid}')):
        res = get_data_from_wdj(name)
        local_list.append(res)
        if sn % 100 == 99:
            print(f'{len(local_list)} rows save to file,  sn:{sn}')
            pd.DataFrame(local_list).to_hdf(file, 'wdj', mode='w')

    pd.DataFrame(local_list).to_hdf(file, 'wdj',mode='w')

    print(f'{len(local_list)} res save to file:{file}')




from multiprocessing import Process



if __name__ == '__main__':
    name_list_len = len(name_list)
    thread_num = 10
    step = name_list_len//thread_num
    for partition_sn in range(thread_num):
        begin , end = step*partition_sn, step*(partition_sn+1)
        if partition_sn ==thread_num-1:
            end =  name_list_len
        print(f'\nthread:{partition_sn}, begin:{begin},end:{end}, total:{name_list_len}')
        p = Process(target=process_name_list, args=(name_list[begin:end],), name=f'p{partition_sn}')
        p.start()




""""
http://so.cr173.com/search/d/%E5%BE%AE%E4%BF%A1_all_rank.html
https://www.baidu.com/s?wd=埃达之光%20安卓
"""