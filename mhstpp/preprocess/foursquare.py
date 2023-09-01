# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import pickle
import os

import numpy as np
import copy
import time
import datetime

# 1. User ID (anonymized)
# 2. Venue ID (Foursquare)
# 3. Venue category ID (Foursquare)
# 4. Venue category name (Fousquare)
# 5. Latitude
# 6. Longitude
# 7. Timezone offset in minutes (The offset in minutes between when this check-in occurred and the same time in UTC)
# 8. UTC time
#

def generate_data(user_count, top_n_item, min_length, max_length, data, BASE_DIR, DATA_SOURCE, small_data=False):
    tr_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                            'tr-user-item-time-top{}-min{}-max{}-{}'.format(top_n_item, min_length,
                                                                                            max_length,
                                                                                            NORM_METHOD) + '.lst')
    # next item
    te_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                'te-user-old-item-time-top{}-min{}-max{}-{}'.format(top_n_item, min_length,
                                                                                                    max_length,
                                                                                                    NORM_METHOD) + '.lst')
    # next new item
    te_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                'te-user-new-item-time-top{}-min{}-max{}-{}'.format(top_n_item, min_length,
                                                                                                    max_length,
                                                                                                    NORM_METHOD) + '.lst')

    # # # normal baseline
    # # tr_baseline_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
    # #                                                  'baseline_tr_top{}.lst'.format(top_n_item))
    # # te_baseline_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
    # #                                                      'baseline_te_old_top{}.lst'.format(top_n_item))
    # # te_baseline_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
    # #                                                      'baseline_te_new_top{}.lst'.format(top_n_item))

    # # # lstm baseline
    # # tr_lstm_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'lstm_tr_top{}.lst'.format(top_n_item))
    # # te_lstm_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'lstm_te_old_top{}.lst'.format(top_n_item))
    # # te_lstm_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'lstm_te_new_top{}.lst'.format(top_n_item))

    # # STAN baseline
    # tr_stan_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'stan_tr_top{}.lst'.format(top_n_item))
    # # te_stan_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'stan_te_old_top{}.lst'.format(top_n_item))
    # # te_stan_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'stan_te_new_top{}.lst'.format(top_n_item))
    
    # # EEDN baseline
    # tr_EEDN_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'EEDN_tr_top{}.lst'.format(top_n_item))
    # te_EEDN_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'EEDN_te_top{}.lst'.format(top_n_item))
    # # te_EEDN_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'EEDN_te_old_top{}.lst'.format(top_n_item))
    # # te_stan_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'stan_te_new_top{}.lst'.format(top_n_item))
    
    # # HGARN baseline
    # tr_HGARN_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'HGARN_tr_top{}.lst'.format(top_n_item))
    # # te_HGARN_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'HGARN_te_top{}.lst'.format(top_n_item))
    # # te_HGARN_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'EEDN_te_old_top{}.lst'.format(top_n_item))
    # # te_HGARN_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'stan_te_new_top{}.lst'.format(top_n_item))
    
    index2item_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                   'foursquare_index2item_topi' + str(top_n_item) + '_topu' + str(user_count))
    item2index_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                   'foursquare_item2index_topi' + str(top_n_item) + '_topu' + str(user_count))

    out_tr_uit = open(tr_user_item_time_record, 'w', encoding='utf-8')
    out_te_old_uit = open(te_user_old_item_time_record, 'w', encoding='utf-8')
    out_te_new_uit = open(te_user_new_item_time_record, 'w', encoding='utf-8')

    # # baseline
    # out_tr_baseline_uit = open(tr_baseline_user_item_time_record, 'w', encoding='utf-8')
    # # 首行
    # out_tr_baseline_uit.write(str(user_count) + ", " + str(top_n_item) + '\n')

    # out_te_baseline_old_uit = open(te_baseline_user_old_item_time_record, 'w', encoding='utf-8')
    # out_te_baseline_new_uit = open(te_baseline_user_new_item_time_record, 'w', encoding='utf-8')

    # # lstm baseline
    # out_tr_lstm_uit = open(tr_lstm_user_item_time_record, 'w', encoding='utf-8')
    # # 首行
    # out_tr_lstm_uit.write(str(user_count) + ", " + str(top_n_item) + '\n')

    # out_te_lstm_old_uit = open(te_lstm_user_old_item_time_record, 'w', encoding='utf-8')
    # out_te_lstm_new_uit = open(te_lstm_user_new_item_time_record, 'w', encoding='utf-8')


    # # stan baseline
    # stan_item_poi = os.path.join(BASE_DIR, DATA_SOURCE, 'stan_poi_top{}.lst'.format(top_n_item))
    # out_tr_stan_uit = open(tr_stan_user_item_time_record, 'w', encoding='utf-8')
    # # 首行
    # out_tr_stan_uit.write(str(user_count) + ", " + str(top_n_item) + '\n')

    # # out_te_stan_old_uit = open(te_stan_user_old_item_time_record, 'w', encoding='utf-8')
    # # out_te_stan_new_uit = open(te_stan_user_new_item_time_record, 'w', encoding='utf-8')
    
    # # EEDN baseline
    # EEDN_item_poi = os.path.join(BASE_DIR, DATA_SOURCE, 'EEDN_poi_top{}.lst'.format(top_n_item))
    # out_tr_EEDN_uit = open(tr_EEDN_user_item_time_record, 'w', encoding='utf-8')
    # # 首行
    # out_tr_EEDN_uit.write(str(user_count) + ", " + str(top_n_item) + '\n')
    # out_te_EEDN_uit = open(te_EEDN_user_item_time_record, 'w', encoding='utf-8')

    # # HGARN baseline
    # HGARN_item_poi = os.path.join(BASE_DIR, DATA_SOURCE, 'HGARN_poi_top{}.lst'.format(top_n_item))
    # out_tr_HGARN_uit = open(tr_HGARN_user_item_time_record, 'w', encoding='utf-8')
    # # 首行
    # out_tr_HGARN_uit.write(str(user_count) + ", " + str(top_n_item) + '\n')
    # # out_te_HGARN_uit = open(te_HGARN_user_item_time_record, 'w', encoding='utf-8')

    # # out_te_stan_old_uit = open(te_stan_user_old_item_time_record, 'w', encoding='utf-8')
    # # out_te_stan_new_uit = open(te_stan_user_new_item_time_record, 'w', encoding='utf-8')
    
    
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
   
        # ##### ======== stan 模型需要poi地点信息
        # poi = data[data['item_id'].isin(item_index2item)].iloc[:,[1, 4, 5]].drop_duplicates()
        # pickle.dump(poi, open(stan_item_poi, 'wb'))
        # #####
        
        # ##### ======== EEDN 模型需要每个poi被访问的次数
        # poi_count = []
        # poi_filter = data[data['item_id'].isin(item_index2item)]
        # for l in poi.iloc[:,0]:
        #     poi_count.append(len(poi_filter[l==poi_filter['item_id']]))   
        # poi['count'] = poi_count
        # pickle.dump(poi, open(EEDN_item_poi, 'wb'))
        # #####
        
        new_user_index2item = [('user_' + str(x)) for x in user_index2item]  # 区分用户和item
        index2item = item_index2item + new_user_index2item  # user和item都放在index2item里面
        print('index2item size is: {}'.format(len(index2item)))

        print('Most common item is "%s":%d' % (index2item[0], sorted_item_series[index2item[0]]))
        # print('Most active user is "%s":%d' % (index2item[top_n_item], sorted_user_series[index2item[top_n_item]]))

        print('build item2index')
        item2index = dict((v, i) for i, v in enumerate(index2item))
        # # -----------SAE-NAD --------------
        # poi_data = data.loc[data['item_id'].isin(item_index2item),['item_id', 'latitude', 'longitude']].drop_duplicates(subset=['item_id'], keep='first')
        # for index, row in poi_data.iterrows():
        #     poi_data.loc[index, 'item_id'] = item2index[row['item_id']]
        # poi_data.to_csv('poi_top_'+ str(top_n_item) + '.csv', header=False)
        
        pickle.dump(index2item, open(index2item_path, 'wb'))
        pickle.dump(item2index, open(item2index_path, 'wb'))

    print('start loop')
    count = 0
    valid_user_count = 0
    user_group = data.groupby(['user_id'])
    total = len(user_group)
    # short sequence comes first
    for user_id, length in user_group.size().sort_values().iteritems():
        count += 1
        if ('user_' + str(user_id)) not in item2index:
            continue
        if count % 100 == 0:
            print("=====count %d/%d======" % (count, total))
            print('%s %d' % (user_id, length))

        # oldest data comes first
        # user_data = user_group.get_group(user_id).sort_values(by='timestamp')
        temp_user_data = user_group.get_group(user_id)
        old_time_seq = copy.deepcopy(pd.to_datetime(temp_user_data['timestamp']))
        temp_user_data.loc[:, 'timestamp_new'] = old_time_seq.values
        user_data = temp_user_data.sort_values(by='timestamp_new')
        # user_data = user_data[user['tranname'].notnull()]
        # user_data = user_data[user['tranname'].notnull()]
        paper_seq = user_data['item_id']
        
        # ##### ------ HGARN模型需要
        # item_cat_id_seq = user_data['item_cat_id']
        # item_cat_name_seq = user_data['item_cat_name']
        # ##### 
        
        # time_seq = user_data['timestamp']
        loc_lat_seq = user_data['latitude']
        loc_lon_seq = user_data['longitude']
        time_seq = user_data['timestamp_new']
        # filter the null data.
        paper_seq = paper_seq[paper_seq.notnull()]
        time_seq = time_seq[time_seq.notnull()]
        
        # ##### ------ HGARN模型需要
        # item_cat_id_seq = item_cat_id_seq[item_cat_id_seq.notnull()]
        # item_cat_name_seq = item_cat_name_seq[item_cat_name_seq.notnull()]
        # #####
        
        loc_lat_seq = loc_lat_seq[loc_lat_seq.notnull()]
        loc_lon_seq = loc_lon_seq[loc_lon_seq.notnull()]
        # calculate the difference between adjacent items. -1 means using t[i] = t[i] - t[i+1]
        delta_time = pd.to_datetime(time_seq).diff(-1).astype('timedelta64[s]') * -1
        # map music to index
        item_seq = paper_seq.apply(lambda x: (item2index[x]) if pd.notnull(x) and x in item2index else -1).tolist()
        delta_time = delta_time.tolist()

        delta_time[-1] = 0

        # temp_log_delta_time = np.log(np.array(delta_time) + 1.0 + 1e-6)  # 不写底数时默认以e为底
        # min_log_delta = temp_log_delta_time.min()
        # max_log_delta = temp_log_delta_time.max()
        # mean_log_delta = temp_log_delta_time.mean()
        # std_log_delta = temp_log_delta_time.std()
        # print("temp_log_delta_time min: {}, max: {}, mean: {}, std: {}".format(min_log_delta, max_log_delta,
        #                                                                        mean_log_delta, std_log_delta))
        # # 归一化
        # temp_delta_time = np.array(delta_time)
        # min_delta = temp_delta_time.min()
        # max_delta = temp_delta_time.max()
        # mean_delta = temp_delta_time.mean()
        # std_delta = temp_delta_time.std()
        # print("temp_delta_time min: {}, max: {}, mean: {}, std: {}".format(min_delta, max_delta, mean_delta, std_delta))

        # temp_delta_time2 = np.array(delta_time)/3600
        # min_delta = temp_delta_time2.min()
        # max_delta = temp_delta_time2.max()
        # mean_delta = temp_delta_time2.mean()
        # std_delta = temp_delta_time2.std()
        # print("temp_delta_time min: {}, max: {}, mean: {}, std: {}".format(min_delta, max_delta, mean_delta, std_delta))

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
        
        
        # #### -----------stan----------
        # time_seq = pd.to_datetime(time_seq) 
        # time_minutes_seq = (time_seq - time_seq.min()).dt.total_seconds() // 60
        # time_minutes_seq = np.array(time_minutes_seq)
        # #####
        
        # #### --------HGARN ---------------
        # HGARN_time = np.array(time_seq)
        # HGARN_item_cat_id = np.array(item_cat_id_seq)
        # HGARN_item_cat_name = np.array(item_cat_name_seq)
        # time_struct = time.strptime(time_seq, "%a %b %d %H:%M:%S +0000 %Y")
        # # print(f'time_struct: {time_struct}')
        # calendar_date = datetime.date(time_struct.tm_year, time_struct.tm_mon, time_struct.tm_mday).isocalendar()
        # # print(f'calendar_date: {calendar_date}')
        # # current_week = f'{calendar_date.year}-{calendar_date.week}'
        # current_week = f'{calendar_date[0]}-{calendar_date[1]}'
        # # Encode time
        # #   tim_code = [time_struct.tm_wday+1, time_struct.tm_hour*2+int(time_struct.tm_min/30)+1 ]    # week(1~7), hours(1~48) 
        # tim_code = [time_struct.tm_wday+1, int(time_struct.tm_hour/2)+1] 

        
        # 过滤-1的item
        new_item_seq = []
        HGARN_time_seq = []
        HGARN_item_cat_id_seq = []
        HGARN_item_cat_name_seq = []
        new_time_seq = []
        new_time_accumulate = []
        new_lat_seq = []
        new_lon_seq = []
        valid_count = 0
        for i in range(len(item_seq)):  # 过滤掉 -1 的item
            if item_seq[i] != -1:
                # HGARN_item_cat_id_seq.append(HGARN_item_cat_id[i])
                # HGARN_item_cat_name_seq.append(HGARN_item_cat_name[i])
                new_item_seq.append(item_seq[i])
                # HGARN_time_seq.append(HGARN_time[i])
                # new_time_seq.append(time_minutes_seq[i])
                new_time_accumulate.append(time_accumulate[i])
                new_lat_seq.append(loc_lat_seq[i])
                new_lon_seq.append(loc_lon_seq[i])
                valid_count += 1
            if valid_count >= max_length:
                break
        
        # ##### ======== HGARN 模型需要poi地点信息
        # poi = pd.DataFrame({'pid': new_item_seq, 'lat': new_lat_seq,'lon': new_lon_seq}).drop_duplicates()
        # pickle.dump(poi, open(HGARN_item_poi, 'wb'))
        # #####
        
        
        if len(new_item_seq) < min_length:  # 跳过过滤之后的交互记录少于min_length的用户
            continue
        else:
            valid_user_count += 1
            user_index = item2index['user_' + str(user_id)]
            # baseline的用户index从0开始
            user_index_baseline = user_index - top_n_item
            # 在 hash() 对对象使用时，所得的结果不仅和对象的内容有关，还和对象的 id()，也就是内存地址有关。
            index_hash_remaining = user_index % 10
            # 20%用户的前50%序列训练，后50%作为测试（包括next item和next new item）；
            if index_hash_remaining < 2:
                half_index = int(len(new_item_seq) / 2)
                for i in range(half_index):  # 前50%序列，训练
                    out_tr_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                    
                    # # stan baseline
                    # out_tr_stan_uit.write(
                    #     str(user_index_baseline) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_seq[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                    
                    # # EEDN baseline
                    # out_tr_EEDN_uit.write(
                    #     str(user_index_baseline) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_seq[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                    
                    # # HGARN baseline
                    # out_tr_HGARN_uit.write(
                    #     str(user_index_baseline) + '\t' + str(new_item_seq[i]) + '\t' + str(HGARN_item_cat_id_seq[i]) + '\t' + str(HGARN_item_cat_name_seq[i])+ '\t' + str(HGARN_time_seq[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                    
                for i in range(half_index, int(len(new_item_seq))):  # 后50%序列，测试
                    out_te_old_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                    out_te_new_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')

                #     # stan baseline
                #     out_tr_stan_uit.write(
                #         str(user_index_baseline) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_seq[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                    
                #     # EEDN baseline
                #     out_te_EEDN_uit.write(
                #         str(user_index_baseline) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_seq[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                    
                #     # HGARN baseline
                #     out_tr_HGARN_uit.write(
                #         str(user_index_baseline) + '\t' + str(new_item_seq[i]) + '\t' + str(HGARN_item_cat_id_seq[i]) + '\t' + str(HGARN_item_cat_name_seq[i]) + '\t' + str(HGARN_time_seq[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                # # # baseline
                # out_tr_baseline_uit.write(str(user_index_baseline) + ',')
                # out_tr_baseline_uit.write(':'.join(str(x) for x in new_item_seq[:half_index]) + '\n')

                # out_te_baseline_old_uit.write(str(user_index_baseline) + ',')
                # out_te_baseline_old_uit.write(':'.join(str(x) for x in new_item_seq[half_index:]) + '\n')

                # out_te_baseline_new_uit.write(str(user_index_baseline) + ',')
                # out_te_baseline_new_uit.write(':'.join(str(x) for x in new_item_seq[half_index:]) + '\n')

                # # lstm baseline
                # out_tr_lstm_uit.write(str(user_index_baseline) + ',')
                # out_tr_lstm_uit.write(':'.join(str(x) for x in new_item_seq[:half_index]) + '\n')

                # out_te_lstm_old_uit.write(str(user_index_baseline) + ',')
                # out_te_lstm_old_uit.write(':'.join(str(x) for x in new_item_seq) + '\n')

                # out_te_lstm_new_uit.write(str(user_index_baseline) + ',')
                # out_te_lstm_new_uit.write(':'.join(str(x) for x in new_item_seq) + '\n')

            else:  # 80%用户的所有序列作为训练
                for i in range(len(new_item_seq)):
                    out_tr_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')

                    # # stan baseline
                    # out_tr_stan_uit.write(
                    #     str(user_index_baseline) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_seq[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                    
                    # # EEDN baseline
                    # out_tr_EEDN_uit.write(
                    #     str(user_index_baseline) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_seq[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                   
                    # # HGARN baseline
                    # out_tr_HGARN_uit.write(
                    #     str(user_index_baseline) + '\t' + str(new_item_seq[i]) + '\t' + str(HGARN_item_cat_id_seq[i]) + '\t' + str(HGARN_item_cat_name_seq[i]) + '\t' + str(HGARN_time_seq[i]) + '\t' + str(new_lat_seq[i]) + '\t' + str(new_lon_seq[i]) + '\n')
                
                # # baseline
                # out_tr_baseline_uit.write(str(user_index_baseline) + ',')
                # out_tr_baseline_uit.write(':'.join(str(x) for x in new_item_seq) + '\n')
                # # lstm baseline
                # out_tr_lstm_uit.write(str(user_index_baseline) + ',')
                # out_tr_lstm_uit.write(':'.join(str(x) for x in new_item_seq) + '\n')

    print("valid_user_count is: {}".format(valid_user_count))
    out_tr_uit.close()
    out_te_old_uit.close()
    out_te_new_uit.close()
    # out_tr_baseline_uit.close()
    # out_te_baseline_old_uit.close()
    # out_te_baseline_new_uit.close()
    # out_tr_lstm_uit.close()
    # out_te_lstm_old_uit.close()
    # out_te_lstm_new_uit.close()
    # out_tr_stan_uit.close()
    # # out_te_stan_old_uit.close()
    # # out_te_stan_new_uit.close()
    # out_tr_EEDN_uit.close()
    # out_te_EEDN_uit.close()
    # out_tr_HGARN_uit.close()
    
    
if __name__ == '__main__':
    BASE_DIR = ''
    DATA_SOURCE = 'foursquare/NYC'

    # NORM_METHOD = 'origin'

    # NORM_METHOD = 'log'
    # (x-min)/(max-min)
    # NORM_METHOD = 'mm'
    # /3600
    NORM_METHOD = 'hour'

    # top_n = 5000  # 频率排前topn的item
    # top_n_item = 10000  # 频率排前topn的item
    top_n_item_list = [9000, 8000, 7000, 6000, 5000]
    # top_n_item_list = [10000, 15000, 20000, 25000, 30000]  # 频率排前topn的item
    min_length = 100  # more than
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

    # The file dataset_TSMC2014_NYC.txt contains 227428 check-ins in New York city.
    # The file dataset_TSMC2014_TKY.txt contains 573703 check-ins in Tokyo.
    # path = os.path.join(BASE_DIR, DATA_SOURCE, 'dataset_TSMC2014_TKY.txt')
    path = os.path.join(BASE_DIR, DATA_SOURCE, 'dataset_TSMC2014_NYC.txt')
    # filter bad lines in original file
    print("start reading csv")
    data = pd.read_csv(path, sep='\t',
                       error_bad_lines=False,
                       header=None,
                       names=['user_id', 'item_id', 'item_cat_id', 'item_cat_name', 'latitude',
                              'longitude', 'time_zone_offset', 'timestamp'],
                       quotechar=None, quoting=3, encoding='ISO-8859-1')
    print("finish reading csv")

    data =  data.dropna()
    for top_n_item in top_n_item_list:
        print("starting processing for top_n_item = {}".format(top_n_item))
        generate_data(user_count, top_n_item, min_length, max_length, data, BASE_DIR, DATA_SOURCE, small_data=False)

