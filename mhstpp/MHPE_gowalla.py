# -*- coding: utf-8 -*-
import os
import logging
import config
import torch

FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
device = torch.device("cuda:3")
# device = 'cpu'
# DID = 1
# os.environ["CUDA_VISIBLE_DEVICES"] = config.MHPE_gowalla_cuda
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pandas as pd

if __name__ == '__main__':
    # 文件格式： 节点1，节点2，时间
    # last fm，user=992，music=5000

    NORM_METHOD = 'hour'
    print('NORM_METHOD: {}'.format(NORM_METHOD))

    data_index = 2  # gowalla
    dataset = config.dataset[data_index]

    min_length = config.min_length[data_index]  # more than
    max_length = config.max_length[data_index]  # less than

    top_n_user = config.user_count_list[data_index]

    for top_n_freq in [30000, 25000, 20000, 15000, 10000]:
    # for top_n_freq in config.top_n_item_list:
        # logging.info('p_lambda = long_pref_weight * p_mu + short_pref_weight * (p_alpha * (torch.exp(torch.neg(delta) * d_time) * h_time_mask + torch.exp(torch.neg(beta) * d_dict) * h_time_mask)).sum(dim=1)')
        logging.info('+++++++++++++ start top_n_item is: {} ++++++++++++++++'.format(top_n_freq))
        logging.info('+++++++++++++ start top_n_item is: {} ++++++++++++++++'.format(top_n_freq))
        logging.info('+++++++++++++ start top_n_item is: {} ++++++++++++++++'.format(top_n_freq))
        train_path = './preprocess/{}/tr-user-item-time-top{}-min{}-max{}-{}.lst'.format(dataset, top_n_freq,
                                                                                         min_length,
                                                                                         max_length,
                                                                                         NORM_METHOD)
        test_old_path = './preprocess/{}/te-user-old-item-time-top{}-min{}-max{}-{}.lst'.format(
            dataset, top_n_freq, min_length, max_length, NORM_METHOD)
        test_new_path = './preprocess/{}/te-user-new-item-time-top{}-min{}-max{}-{}.lst'.format(
            dataset, top_n_freq, min_length, max_length, NORM_METHOD)

        # htne = HTSER_a(train_path, test_old_path, test_new_path, emb_size=256, neg_size=5, hist_len=3, user_count=1000,
        #     item_count=top_n_freq, directed=True, learning_rate=0.0003, decay=0.001, batch_size=256, test_and_save_step=20,
        #     epoch_num=300, top_n=30, use_hist_attention=False, use_user_pref_attention=True, num_workers=8)
        poi_data_path = '{}/poi_top_{}.csv'.format(dataset, top_n_freq)
        poi_data = pd.read_csv(poi_data_path, header=None)
        htne = config.MHPE_gowalla_model.HTSER_a(train_path, test_old_path, test_new_path,
                                                emb_size=config.MHPE_gowalla_emb_size,
                                                neg_size=config.MHPE_gowalla_neg_size,
                                                hist_len=config.MHPE_gowalla_hist_len,
                                                user_count=top_n_user,
                                                item_count=top_n_freq, directed=True,
                                                learning_rate=config.MHPE_gowalla_learning_rate,
                                                decay=config.MHPE_gowalla_decay,
                                                batch_size=config.MHPE_gowalla_batch_size,
                                                test_and_save_step=config.MHPE_gowalla_test_and_save_step,
                                                epoch_num=config.MHPE_gowalla_epoch_num, top_n=30,
                                                use_hist_attention=False,
                                                use_user_pref_attention=True, num_workers=8,poi_data=poi_data,device=device)
        MAX_1 = 0 
        MAX_5 = 0 
        MAX_10 = 0 
        MAX_20 = 0 
        htne.train()

        logging.info('------------- end top_n_item is: {} -----------------'.format(top_n_freq))
        logging.info('------------- end top_n_item is: {} -----------------'.format(top_n_freq))
        logging.info('------------- end top_n_item is: {} -----------------'.format(top_n_freq))
        
        


