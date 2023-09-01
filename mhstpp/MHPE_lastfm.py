# -*- coding: utf-8 -*-
import os
import logging
import config

FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

# DID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = config.MHPE_lastfm_cuda

if __name__ == '__main__':
    # 文件格式： 节点1，节点2，时间
    # last fm，user=992，music=5000

    NORM_METHOD = 'hour'
    print('NORM_METHOD: {}'.format(NORM_METHOD))

    data_index = 0  # lastfm
    dataset = config.dataset[data_index]

    min_length = config.min_length[data_index]  # more than
    max_length = config.max_length[data_index]  # less than

    top_n_user = config.user_count_list[data_index]

    # for top_n_freq in [9000, 8000, 7000, 6000, 5000]:
    for top_n_freq in config.top_n_item_list:
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

        # htne = HTSER_a(train_path, test_old_path, test_new_path, emb_size=256, neg_size=5, hist_len=3, user_count=992,
        #                item_count=top_n_freq, directed=True, learning_rate=0.001, decay=0.003, batch_size=512, test_and_save_step=20,
        #                epoch_num=300, top_n=30, use_hist_attention=False, use_user_pref_attention=True, num_workers=8)

        # htne = config.MHPE_lastfm_model.HTSER_a(train_path, test_old_path, test_new_path, emb_size=256, neg_size=5, hist_len=3, user_count=992,
        #                item_count=top_n_freq, directed=True, learning_rate=0.001, decay=0.003, batch_size=512, test_and_save_step=20,
        #                epoch_num=300, top_n=30, use_hist_attention=False, use_user_pref_attention=True, num_workers=8)

        htne = config.MHPE_lastfm_model.HTSER_a(train_path, test_old_path, test_new_path,
                                                emb_size=config.MHPE_lastfm_emb_size,
                                                neg_size=config.MHPE_lastfm_neg_size,
                                                hist_len=config.MHPE_lastfm_hist_len,
                                                user_count=top_n_user,
                                                item_count=top_n_freq, directed=True,
                                                learning_rate=config.MHPE_lastfm_learning_rate,
                                                decay=config.MHPE_lastfm_decay,
                                                batch_size=config.MHPE_lastfm_batch_size,
                                                test_and_save_step=config.MHPE_lastfm_test_and_save_step,
                                                epoch_num=config.MHPE_lastfm_epoch_num, top_n=30,
                                                use_hist_attention=False,
                                                use_user_pref_attention=True, num_workers=8)

        htne.train()

        logging.info('------------- end top_n_item is: {} -----------------'.format(top_n_freq))
        logging.info('------------- end top_n_item is: {} -----------------'.format(top_n_freq))
        logging.info('------------- end top_n_item is: {} -----------------'.format(top_n_freq))
