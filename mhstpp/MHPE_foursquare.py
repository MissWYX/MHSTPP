# -*- coding: utf-8 -*-
import os
import logging
import config
import torch
FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

# DID = 1
# os.environ["CUDA_VISIBLE_DEVICES"] = config.MHPE_foursquare_cuda
device = torch.device("cuda:3")
# device = 'cpu'
import pandas as pd
print(torch.cuda.is_available())
import torch
# os.system('nvidia-smi')
# num_gpus = torch.cuda.device_count()

# if num_gpus > 0:
#     print(f"Total available GPUs: {num_gpus}")
#     for gpu_idx in range(num_gpus):
#         print(f"GPU {gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}")
# else:
#     print("No GPUs available on this system.")

# # 检查CUDA是否可用
# if torch.cuda.is_available():
#     # 获取当前GPU设备
#     device = torch.device("cuda:3")
#     print(torch.__version__)
#     # 获取CUDA内存统计信息
#     memory_stats = torch.cuda.memory_stats(device=device)

#     # 获取可用内存和总内存大小（以字节为单位）
#     print(memory_stats.keys())
#     free_memory = memory_stats["allocated_bytes.all.current"]
#     total_memory = memory_stats["allocated_bytes.all.allocated"]

#     # 将字节转换为MB
#     free_memory_mb = free_memory / (1024 ** 2)
#     total_memory_mb = total_memory / (1024 ** 2)

#     print(f"Free GPU Memory: {free_memory_mb:.2f} MB")
#     print(f"Total GPU Memory: {total_memory_mb:.2f} MB")
# else:
#     print("CUDA is not available.")

if __name__ == '__main__':
    # 文件格式： 节点1，节点2，时间
    # last fm，user=992，music=5000

    NORM_METHOD = 'hour'
    print('NORM_METHOD: {}'.format(NORM_METHOD))

    data_index = 3  # foursquare
    dataset = config.dataset[data_index]

    min_length = config.min_length[data_index]  # more than
    max_length = config.max_length[data_index]  # less than

    # top_n_user = config.user_count_list[data_index]
    top_n_user = 2000
    
    for top_n_freq in [9000, 8000, 7000, 6000, 5000]:
    # for top_n_freq in config.top_n_item_list:
        # logging.info('p_lambda = long_pref_weight * p_mu + short_pref_weight * (p_alpha * (torch.exp(torch.neg(delta) * d_time) * h_time_mask + torch.exp(torch.neg(beta) * d_dict) * h_time_mask)).sum(dim=1)')
        logging.info('+++++++++++++ start top_n_item is: {} ++++++++++++++++'.format(top_n_freq))
        logging.info('+++++++++++++ start top_n_item is: {} ++++++++++++++++'.format(top_n_freq))
        logging.info('+++++++++++++ start top_n_item is: {} ++++++++++++++++'.format(top_n_freq))
        train_path = './preprocess/{}/TKY/tr-user-item-time-top{}-min{}-max{}-{}.lst'.format(dataset, top_n_freq,
                                                                                         min_length,
                                                                                         max_length,
                                                                                         NORM_METHOD)
        test_old_path = './preprocess/{}/TKY/te-user-old-item-time-top{}-min{}-max{}-{}.lst'.format(
            dataset, top_n_freq, min_length, max_length, NORM_METHOD)
        test_new_path = './preprocess/{}/TKY/te-user-new-item-time-top{}-min{}-max{}-{}.lst'.format(
            dataset, top_n_freq, min_length, max_length, NORM_METHOD)
        
        poi_data_path = '{}/TKY/poi_top_{}.csv'.format(dataset, top_n_freq)
        poi_data = pd.read_csv(poi_data_path, header=None)
        # htne = HTSER_a(train_path, test_old_path, test_new_path, emb_size=32, neg_size=5, hist_len=3, user_count=992,
        #    item_count=top_n_freq, directed=True, learning_rate=0.001, decay=0.003, batch_size=512, test_and_save_step=20,
        #    epoch_num=200, top_n=30, use_hist_attention=True, use_user_pref_attention=True, num_workers=8)

        htne = config.MHPE_foursquare_model.HTSER_a(train_path, test_old_path, test_new_path,
                                                 emb_size=config.MHPE_foursquare_emb_size,
                                                 neg_size=config.MHPE_foursquare_neg_size,
                                                 hist_len=config.MHPE_foursquare_hist_len,
                                                 user_count=top_n_user,
                                                 item_count=top_n_freq, directed=True,
                                                 learning_rate=config.MHPE_foursquare_learning_rate,
                                                 decay=config.MHPE_foursquare_decay,
                                                 batch_size=config.MHPE_foursquare_batch_size,
                                                 test_and_save_step=config.MHPE_foursquare_test_and_save_step,
                                                 epoch_num=config.MHPE_foursquare_epoch_num, top_n=30,
                                                 use_hist_attention=False,
                                                 use_user_pref_attention=True, num_workers=8,poi_data=poi_data,device=device)

        
        htne.train()

        logging.info('------------- end top_n_item is: {} -----------------'.format(top_n_freq))
        logging.info('------------- end top_n_item is: {} -----------------'.format(top_n_freq))
        logging.info('------------- end top_n_item is: {} -----------------'.format(top_n_freq))
        break
