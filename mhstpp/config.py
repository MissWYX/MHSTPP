# -*- coding: utf-8 -*-
# import Model_origin
import Model_new_user_attention3 as Model
# import Model_new_user_attention3_movielens as Model_Movielens
# import Model_origin_only_relu
# import Model_origin_only_soft
# import Model_sqrt_dur_hiera_long_short_pref_new_dur_pos_attention_euc_dist
# import Model_pow_dur_hiera_long_short_pref_new_dur_pos_attention_euc_dist

# common
# dataset = ['last_music', 'citeulike', 'gowalla', 'foursquare']
# min_length = [100, 20, 100, 100]  # more than
# max_length = [1000, 500, 500, 500]  # less than
# user_count_list = [992, 3000, 2000, 2000]
# top_n_item_list = [30000, 25000, 20000, 15000, 10000]

dataset = ['last_music', 'movielens', 'gowalla', 'foursquare']
min_length = [100, 100, 100, 100]  # more than
max_length = [1000, 500, 500, 500]  # less than
# user_count_list = [992, 5000, 2000, 2000]
user_count_list = [992, 3000, 2000, 2000]
top_n_item_list = [30000, 25000, 20000, 15000, 10000]
top_n_item_list_movielens = [25000, 20000, 15000, 10000, 5000]

hist_length_list = [1, 2, 3, 4, 5, 6, 7]

# =======================================
# lastfm
MHPE_lastfm_model = Model
MHPE_lastfm_emb_size = 256
# MHPE_lastfm_neg_size = 10
MHPE_lastfm_neg_size = 5
MHPE_lastfm_hist_len = 3
MHPE_lastfm_cuda = "0"
# MHPE_lastfm_learning_rate = 0.0003
MHPE_lastfm_learning_rate = 0.001
MHPE_lastfm_decay = 0.01
MHPE_lastfm_batch_size = 1024
# MHPE_lastfm_batch_size = 512
MHPE_lastfm_test_and_save_step = 20
# MHPE_lastfm_epoch_num = 100
MHPE_lastfm_epoch_num = 200

# =======================================
# movielens
# MHPE_movielens_model = Model
# MHPE_movielens_model = Model_Movielens
# MHPE_movielens_emb_size = 256
# MHPE_movielens_neg_size = 5
# MHPE_movielens_hist_len = 3
# MHPE_movielens_cuda = "1"
# # MHPE_movielens_learning_rate = 0.001
# MHPE_movielens_learning_rate = 0.003
# # MHPE_movielens_decay = 0.01
# MHPE_movielens_decay = 0.003
# # MHPE_movielens_batch_size = 1024
# MHPE_movielens_batch_size = 2048
# MHPE_movielens_test_and_save_step = 20
# MHPE_movielens_epoch_num = 200

# =======================================
# gowalla
MHPE_gowalla_model = Model
MHPE_gowalla_emb_size = 64
# MHPE_gowalla_neg_size = 10
MHPE_gowalla_neg_size = 5
MHPE_gowalla_hist_len = 1
MHPE_gowalla_cuda = "3"
# MHPE_gowalla_learning_rate = 0.0003
MHPE_gowalla_learning_rate = 0.001
# MHPE_gowalla_decay = 0.01
MHPE_gowalla_decay = 0.001
MHPE_gowalla_batch_size = 256
MHPE_gowalla_test_and_save_step = 20
# MHPE_gowalla_epoch_num = 100
MHPE_gowalla_epoch_num = 200

# =======================================
# foursquare
MHPE_foursquare_model = Model
MHPE_foursquare_emb_size = 128
# MHPE_foursquare_neg_size = 10
MHPE_foursquare_neg_size = 5
MHPE_foursquare_hist_len = 1
MHPE_foursquare_cuda = "3"
# MHPE_foursquare_learning_rate = 0.0003
MHPE_foursquare_learning_rate = 0.001
MHPE_foursquare_decay = 0.01
MHPE_foursquare_batch_size = 256
MHPE_foursquare_test_and_save_step = 20
# MHPE_foursquare_epoch_num = 100
MHPE_foursquare_epoch_num = 200


# # =======================================
# # citeulike
MHPE_citeulike_model = Model
MHPE_citeulike_emb_size = 256
# MHPE_citeulike_neg_size = 10
MHPE_citeulike_neg_size = 5
MHPE_citeulike_hist_len = 3
MHPE_citeulike_cuda = "0"
MHPE_citeulike_learning_rate = 0.0003
MHPE_citeulike_decay = 0.01
MHPE_citeulike_batch_size = 256
MHPE_citeulike_test_and_save_step = 20
# MHPE_citeulike_epoch_num = 100
MHPE_citeulike_epoch_num = 200