from spider.mi import *
import re
from core.conf import check_type_list

if __name__ == '__main__':
    import sys

    print(sys.argv)

    #max_len = sys.argv[1]

    sys.argv=['a','b']

    final = get_final_feature()
    final.desc_name = final.desc_name.str[:4000]

    file = './input/zip/apptype_train.dat_p2'

    train = final.loc[final.type_id.str.len() > 0]
    train['id'] = 'valxxx' + train['id'].str[6:]
    train.loc[:, ['id', 'type_id', 'desc_name']].to_csv(file, sep='\t', header=None, index=None)
    print(f'save {len(train)} rows to {file} ')

    file='./input/zip/app_desc.dat'
    test = final.loc[final.type_id.str.len()==0]#.loc[final.type_id.str.len()==0]
    test = pd.concat([train, test])
    test.loc[:,['id','desc_name']].to_csv(file, sep='\t',header=None, index=None)
    print(f'save {len(test)} rows to {file} ')





#
# #######################
#     train_list = []
#     for item in check_type_list:
#         if item =='stb':
#             continue
#         final = get_final_feature(item)
#         train = final.loc[final.type_id.str.len() > 0]
#         train['id'] = train['id'].apply(lambda val: item + 'x'*(6-len(item)) + val[6:])
#         train_list.append(train)
#
#     #Stb part
#     stb = pd.read_csv('./input/zip/78_app_desc.dat', sep='\t', header=None)
#     stb.columns = ['id', 'desc_name']
#     tmp = get_train_ph2_index()
#     stb = stb.loc[stb['id'].isin(tmp['id'])]  # .shape
#     stb.head()
#     stb.id = 'stbxxx' + stb.id.str[6:]
#     train_list.append(stb)
#     #Stb_end
#
#     train = pd.concat(train_list)
#     file = f'./input/zip/app_desc.dat'
#
#     train.loc[:, ['id', 'desc_name']].to_csv(file, sep='\t', header=None, index=None)
#     print(f'save to {file} ')

"""
python spider/gen_file.py
"""
