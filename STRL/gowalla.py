# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import pickle
import os

import numpy as np


def generate_data(user_count, top_n_item, min_length, max_length, data, BASE_DIR, DATA_SOURCE, small_data=False):
    
    # STRL baseline
    tr_STRL_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'STRL_tr_top{}.lst'.format(top_n_item))
    tu_STRL_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'STRL_tu_top{}.lst'.format(top_n_item))
    te_STRL_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'STRL_te_top{}.lst'.format(top_n_item))
      
    
    index2item_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                   'gowalla_index2item_topi' + str(top_n_item) + '_topu' + str(user_count))
    item2index_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                   'gowalla_item2index_topi' + str(top_n_item) + '_topu' + str(user_count))


    
    out_tr_STRL_uit = open(tr_STRL_user_item_time_record, 'w', encoding='utf-8')
    out_tu_STRL_uit = open(tu_STRL_user_item_time_record, 'w', encoding='utf-8')
    out_te_STRL_uit = open(te_STRL_user_item_time_record, 'w', encoding='utf-8')
 
    if os.path.exists(index2item_path) and os.path.exists(item2index_path):
        index2item = pickle.load(open(index2item_path, 'rb'))
        item2index = pickle.load(open(item2index_path, 'rb'))
        print('Total item %d' % len(index2item))
    else:
        print('Build index2item')
        sorted_user_series = data.groupby(['user_id']).size().sort_values(ascending=False)
        print('sorted_user_series size is: {}'.format(len(sorted_user_series)))
        # user_index2item = sorted_user_series.keys().tolist()  # 所有用户，按照从高到低排序
        user_index2item = sorted_user_series.head(user_count).keys().tolist()  # 所有用户，按照从高到低排序

        sorted_item_series = data.groupby(['item_id']).size().sort_values(ascending=False)
        print('sorted_item_series size is: {}'.format(len(sorted_item_series)))
        item_index2item = sorted_item_series.head(top_n_item).keys().tolist()  # 只取前 top_n个item
        print('item_index2item size is: {}'.format(len(item_index2item)))
        
        
        
        new_user_index2item = [('user_' + str(x)) for x in user_index2item]  # 区分用户和item
        index2item = item_index2item + new_user_index2item  # user和item都放在index2item里面
        print('index2item size is: {}'.format(len(index2item)))

        print('Most common item is "%s":%d' % (index2item[0], sorted_item_series[index2item[0]]))
        # print('Most active user is "%s":%d' % (index2item[top_n_item], sorted_user_series[index2item[top_n_item]]))

        print('build item2index')
        item2index = dict((v, i) for i, v in enumerate(index2item))
        
        
        # -----------SAE-NAD --------------
        poi_data = data.loc[data['item_id'].isin(item_index2item),['item_id', 'latitude', 'longitude']].drop_duplicates(subset=['item_id'], keep='first')
        for index, row in poi_data.iterrows():
            poi_data.loc[index, 'item_id'] = item2index[row['item_id']]
        poi_data.to_csv('poi_top_'+ str(top_n_item) + '.csv', header=False)
        
        pickle.dump(index2item, open(index2item_path, 'wb'))
        pickle.dump(item2index, open(item2index_path, 'wb'))

    print('start loop')
    count = 0
    valid_user_count = 0
    user_group = data.groupby(['user_id'])
    total = len(user_group) # 107092
    # short sequence comes first
    
    
    
    for user_id, length in user_group.size().sort_values().iteritems():
        count += 1
        if ('user_' + str(user_id)) not in item2index:
            continue
        if count % 100 == 0:
            print("=====count %d/%d======" % (count, total))
            print('%s %d' % (user_id, length))

        # oldest data comes first
        user_data = user_group.get_group(user_id).sort_values(by='timestamp')
        # user_data = user_data[user['tranname'].notnull()]
        paper_seq = user_data['item_id']
        loc_lat_seq = user_data['latitude']
        loc_lon_seq = user_data['longitude']
        time_seq = user_data['timestamp']
        # filter the null data.
        paper_seq = paper_seq[paper_seq.notnull()]
        loc_lat_seq = loc_lat_seq[loc_lat_seq.notnull()]
        loc_lon_seq = loc_lon_seq[loc_lon_seq.notnull()]
        time_seq = time_seq[time_seq.notnull()]
        # calculate the difference between adjacent items. -1 means using t[i] = t[i] - t[i+1]
        delta_time = pd.to_datetime(time_seq).diff(-1).astype('timedelta64[s]') * -1
        # map music to index
        item_seq = paper_seq.apply(lambda x: (item2index[x]) if pd.notnull(x) and x in item2index else -1).tolist()
        delta_time = delta_time.tolist()

        delta_time[-1] = 0

        if NORM_METHOD == 'log':
            # 这里我们使用对数函数来对间隔时间进行缩放
            # + 1.0 + 1e-6  保证结果为正数
            delta_time = np.log(np.array(delta_time) + 1.0 + 1e-6)  # log不写底数时默认以e为底
        
        elif NORM_METHOD == 'mm':
            temp_delta_time = np.array(delta_time)
            min_delta = temp_delta_time.min()
            max_delta = temp_delta_time.max()
            # (temp_delta_time - min_time) / (max_time - min_time)
            delta_time = (np.array(delta_time) - min_delta) / (max_delta - min_delta)
        elif NORM_METHOD == 'hour':
            delta_time = np.array(delta_time) / 3600

        time_accumulate = [0]
        for delta in delta_time[:-1]:
            next_time = time_accumulate[-1] + delta
            time_accumulate.append(next_time)
        
        # # 处理坐标 
        # loc_lat_seq = pd.to_datetime(loc_lat_seq)
        loc_lat_seq = np.array(loc_lat_seq)
        # min_loc_lat = temp_loc_lat_seq.min()
        # max_loc_lat = temp_loc_lat_seq.max()
        # # (temp_delta_time - min_time) / (max_time - min_time)
        # loc_lat_seq = (np.array(loc_lat_seq) - min_loc_lat) / (max_loc_lat - min_loc_lat)
        
        # loc_lon_seq = pd.to_datetime(loc_lon_seq)
        loc_lon_seq = np.array(loc_lon_seq)
        # min_loc_lon = temp_loc_lon_seq.min()
        # max_loc_lon = temp_loc_lon_seq.max()
        # # (temp_delta_time - min_time) / (max_time - min_time)
        # loc_lon_seq = (np.array(loc_lon_seq) - min_loc_lon) / (max_loc_lon - min_loc_lon)
        
        
        #### -----------stan----------
        time_seq = pd.to_datetime(time_seq) 
        time_minutes_seq = (time_seq - time_seq.min()).dt.total_seconds() // 60
        time_minutes_seq = np.array(time_minutes_seq)
        #####
        
        # 过滤-1的item
        new_item_seq = []
        new_time_seq = []
        new_time_accumulate = []
        new_lat_seq = []
        new_lon_seq = []
        valid_count = 0
        for i in range(len(item_seq)):  # 过滤掉 -1 的item
            if item_seq[i] != -1:
                new_item_seq.append(item_seq[i])
                new_time_seq.append(time_minutes_seq[i])
                new_time_accumulate.append(time_accumulate[i])
                new_lat_seq.append(loc_lat_seq[i])
                new_lon_seq.append(loc_lon_seq[i])
                valid_count += 1
            if valid_count >= max_length:
                break

        if len(new_item_seq) < min_length:  # 跳过过滤之后的交互记录少于min_length的用户
            continue
        else:
            valid_user_count += 1
            user_index = item2index['user_' + str(user_id)]
            # baseline的用户index从0开始
            user_index_baseline = user_index - top_n_item  # 因为item2index前部分是item，后部分才是userid
            # 在 hash() 对对象使用时，所得的结果不仅和对象的内容有关，还和对象的 id()，也就是内存地址有关。
            index_hash_remaining = user_index % 10
            if index_hash_remaining < 2:  # 20%用户的前50%序列训练，后50%作为测试（next item和next new item）；
                half_index = int(len(new_item_seq) / 2)
                for i in range(half_index):  # 前50%序列，训练
                    # STRL baseline
                    out_tu_STRL_uit.write(
                        str(user_index_baseline) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_seq[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                for i in range(half_index, int(len(new_item_seq))):  # 后50%序列，测试
                    
                    # STRL baseline
                    out_te_STRL_uit.write(
                        str(user_index_baseline) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_seq[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                    
            else:  # 80%用户的所有序列作为训练
                for i in range(len(new_item_seq)):

                    # SAE_NAD baseline
                    out_tr_STRL_uit.write(
                        str(user_index_baseline) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_seq[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                    
                    

    print("valid_user_count is: {}".format(valid_user_count))

    # out_tr_baseline_uit.close()
    # out_te_baseline_old_uit.close()
    # out_te_baseline_new_uit.close()
    # out_tr_lstm_uit.close()
    # out_te_lstm_old_uit.close()
    # out_te_lstm_new_uit.close()
    # out_tr_stan_uit.close()
    # out_te_stan_old_uit.close()
    # out_te_stan_new_uit.close()
    out_tr_STRL_uit.close()
    out_te_STRL_uit.close()
    out_tu_STRL_uit.close()


if __name__ == '__main__':

    BASE_DIR = ''
    DATA_SOURCE = 'gowalla'

    # NORM_METHOD = 'origin'

    # NORM_METHOD = 'log'
    # (x-min)/(max-min)
    # NORM_METHOD = 'mm'
    # /3600
    NORM_METHOD = 'hour'

    # top_n = 5000  # 频率排前topn的item
    # top_n_item = 10000  # 频率排前topn的item
    min_length = 100  # more than
    top_n_item_list = [10000, 15000, 20000, 25000, 30000]  # 频率排前topn的item
    # max_length = 200  # more than
    # max_length = 1000  # more than
    max_length = 500  # more than

    user_count = 2000  # user_count= 2000,频率排前topn的user

    # 80%用户的所有序列作为训练；
    # 20%用户的前50%序列训练，后50%作为next item测试；
    # 同样的20%用户的前50%序列训练，后50%作为next new item测试；
    # 对于给定的测试用户u（n次交互记录），则会产生n-1个测试用例
    # 对间隔时间进行标准化/归一化
    # 由于每个用户有自己的衰减稀疏delta，所以可以分别对每个用户的数据进行归一化

    # filter bad lines in original file
    # awk -F "|" '{print $1"|"$2"|"$3}' citeulike-origin | uniq > citeulike-origin-filtered
    path = os.path.join(BASE_DIR, DATA_SOURCE, 'Gowalla_totalCheckins.txt')
    print("start reading csv")
    data = pd.read_csv(path, sep='\t',
                       error_bad_lines=False,
                       header=None,
                       names=['user_id', 'timestamp', 'latitude', 'longitude', 'item_id'],
                       quotechar=None, quoting=3)
    print("finish reading csv")

    for top_n_item in top_n_item_list:
        print("starting processing for top_n_item = {}".format(top_n_item))
        generate_data(user_count, top_n_item, min_length, max_length, data, BASE_DIR, DATA_SOURCE, small_data=False)
