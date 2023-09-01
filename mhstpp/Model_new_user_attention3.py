# -*- coding: utf-8 -*-
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import sys
from DataSet import DataSetTrain, DataSetTestNext, DataSetTestNextNew
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('agg')

import os
import torch.nn.functional as F
import logging

FType = torch.FloatTensor
LType = torch.LongTensor

FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

# DID = 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
NORM_METHOD = 'hour'

import math
from math import radians, cos, sin, asin, sqrt
from utils import vincenty_distance, calculate_area

class HTSER_a:
    '''
    有attention机制的HHNER
    Hawkess process based Temporal Sequence Embedding for Recommendtion/Prediction
    Hawkess Process with Hierarchical Attention Network for Recommendtion
    Sequential Recommendation based on Hawkess Process with Attention
    '''

    def __init__(self, file_path_tr, file_path_te_old, file_path_te_new, emb_size=128, neg_size=10,
                 hist_len=2,
                 user_count=992, item_count=5000, directed=True, learning_rate=0.001, decay=0.001, batch_size=1024,
                 test_and_save_step=50, epoch_num=100, top_n=30, sample_time=3, sample_size=100,
                 use_hist_attention=True, use_user_pref_attention=True, num_workers=0, poi_data=None,device=None):
        """
        :param file_path_tr:
        :param emb_size:
        :param neg_size:
        :param hist_len: 历史node的长度，超过则截断
        :param directed: 是否为有向图
        :param learning_rate:
        :param batch_size:
        :param test_and_save_step: 每 save_step步之后，保存一次模型
        :param epoch_num:
        :param use_hist_attention: 是否使用attention机制
        :param num_workers: windows平台 需要 num_workers=0
        """
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.user_count = user_count
        self.item_count = item_count

        self.lr = learning_rate
        self.decay = decay
        self.batch = batch_size
        self.test_and_save_step = test_and_save_step
        self.epochs = epoch_num

        self.top_n = top_n
        self.sample_time = sample_time
        self.sample_size = sample_size

        self.directed = directed
        self.use_hist_attention = use_hist_attention
        self.use_user_pref_attention = use_user_pref_attention

        self.num_workers = num_workers

        self.temp_value1 = 0.0
        self.temp_value2 = 0.0
        
        self.poi_data = poi_data

        self.max_recall = np.zeros(self.top_n)
        self.new_max_recall = np.zeros(self.top_n)
        self.device = device
        self.max_mrr = np.zeros(self.top_n)
        self.new_max_mrr = np.zeros(self.top_n)

        logging.info('emb_size: {}'.format(emb_size))
        logging.info('neg_size: {}'.format(neg_size))
        logging.info('hist_len: {}'.format(hist_len))
        logging.info('user_count: {}'.format(user_count))
        logging.info('item_count: {}'.format(item_count))
        logging.info('lr: {}'.format(learning_rate))
        logging.info('epoch_num: {}'.format(epoch_num))
        logging.info('test_and_save_step: {}'.format(test_and_save_step))
        logging.info('batch: {}'.format(batch_size))
        logging.info('top_n: {}'.format(top_n))
        logging.info('sample_time: {}'.format(sample_time))
        logging.info('sample_size: {}'.format(sample_size))
        logging.info('directed: {}'.format(directed))
        logging.info('use_hist_attention: {}'.format(use_hist_attention))
        logging.info('use_user_pref_attention: {}'.format(use_user_pref_attention))

        self.data_tr = DataSetTrain(file_path_tr, user_count=self.user_count, item_count=self.item_count,
                                    neg_size=self.neg_size, hist_len=self.hist_len, directed=self.directed)
        self.data_te_old = DataSetTestNext(file_path_te_old, user_count=self.user_count, item_count=self.item_count,
                                           hist_len=self.hist_len, user_item_dict=self.data_tr.user_item_dict,
                                           directed=self.directed)
        self.data_te_new = DataSetTestNextNew(file_path_te_new, user_count=self.user_count, item_count=self.item_count,
                                              hist_len=self.hist_len, user_item_dict=self.data_tr.user_item_dict,
                                              directed=self.directed)

        self.node_dim = self.data_tr.get_node_dim()  # 节点个数
        # self.node_emb = torch.tensor(np.random.normal(0, 0.5, size=(self.node_dim, self.emb_size)), dtype=torch.float)
        self.node_emb = torch.tensor(
            np.random.uniform(-0.5 / self.emb_size, 0.5 / self.emb_size, size=(self.node_dim, self.emb_size)),
            dtype=torch.float)
        self.delta = torch.ones(self.node_dim, dtype=torch.float)
        # self.delta = torch.tensor(np.random.normal(1, 1.0 / self.emb_size, size=self.node_dim), dtype=torch.float)
        self.beta = torch.ones(self.node_dim, dtype=torch.float)
        # 初始化方法：He initialization
        self.weight = torch.tensor(
            np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=(self.emb_size, self.emb_size)), dtype=torch.float)
        self.bias = torch.tensor(
            np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=self.emb_size), dtype=torch.float)

        # 用于计算用户长短期兴趣
        # 每个维度的分配不同？
        # self.pref_weight = torch.tensor(
        #     np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=(self.emb_size * 2, self.emb_size)), dtype=torch.float)
        # self.pref_bias = torch.tensor(
        #     np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=self.emb_size), dtype=torch.float)
        # 每个t特征维度的分配相同
        # self.long_pref_weight = torch.tensor(np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=self.emb_size),
        #                                      dtype=torch.float)
        # self.long_pref_bias = torch.tensor(np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=1), dtype=torch.float)
        # self.short_pref_weight = torch.tensor(np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=self.emb_size),
        #                                       dtype=torch.float)
        # self.short_pref_bias = torch.tensor(np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=1),
        #                                     dtype=torch.float)
        # 20201223 模仿AFM进行一些修改
        self.long_short_pref_weight = torch.tensor(
            np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=(2*self.emb_size, 2)),
            dtype=torch.float)
        self.long_short_pref_bias = torch.tensor(np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=2),
                                                 dtype=torch.float)
        # self.hidden_h = torch.tensor(np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=self.emb_size),
        #                              dtype=torch.float)

        # self.short_pref_weight = torch.tensor(
        #     np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=self.emb_size),
        #     dtype=torch.float)
        # self.short_pref_bias = torch.tensor(np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=1),
        #                                     dtype=torch.float)

        if torch.cuda.is_available():
            self.node_emb = self.node_emb.to(self.device)
            self.delta = self.delta.to(self.device)
            self.beta = self.beta.to(self.device)
            self.weight = self.weight.to(self.device)
            self.bias = self.bias.to(self.device)
            self.long_short_pref_weight = self.long_short_pref_weight.to(self.device)
            self.long_short_pref_bias = self.long_short_pref_bias.to(self.device)
            # self.short_pref_weight = self.short_pref_weight.to(self.device)
            # self.short_pref_bias = self.short_pref_bias.to(self.device)
            # self.pref_weight = self.pref_weight.to(self.device)
            # self.pref_bias = self.pref_bias.to(self.device)
            # self.hidden_h = self.hidden_h.to(self.device)
        self.node_emb.requires_grad = True
        self.delta.requires_grad = True
        self.beta.requires_grad = True
        self.weight.requires_grad = True
        self.bias.requires_grad = True
        self.long_short_pref_weight.requires_grad = True
        self.long_short_pref_bias.requires_grad = True
        # self.short_pref_weight.requires_grad = True
        # self.short_pref_bias.requires_grad = True
        # self.pref_weight.requires_grad = True
        # self.pref_bias.requires_grad = True
        # self.hidden_h.requires_grad = True
        # self.opt = Adam(lr=self.lr, params=[self.node_emb, self.delta, self.weight, self.bias])
        self.opt = Adam(lr=self.lr,
                        params=[self.node_emb, self.delta, self.beta, self.weight, self.bias, self.long_short_pref_weight,
                                self.long_short_pref_bias], weight_decay=self.decay)
        self.loss = torch.FloatTensor()

    def forward(self, s_nodes, t_nodes, t_times, t_loc_lat, t_loc_lon, n_nodes, h_nodes, h_times, h_locs_lat, h_locs_lon, h_time_mask):
        '''
        :param s_nodes: source nodes
        :param t_nodes: target nodes
        :param t_times: target nodes的时间戳
        :param n_nodes: negative nodes
        :param h_nodes: historical nodes
        :param h_times: historical nodes的时间戳
        :param h_time_mask:
        :return: p_lambda, n_lambda 分别为目标节点和负采样节点的概率密度
        '''
        batch = s_nodes.size()[0]
        # 如果是torch.view(-1)，则原张量会变成一维的结构。
        # b × d
        s_node_emb = torch.index_select(self.node_emb, 0, s_nodes.view(-1)).view(batch, -1)
        # b × d
        t_node_emb = torch.index_select(self.node_emb, 0, t_nodes.view(-1)).view(batch, -1)
        # h_node_emb: b × h × d
        h_node_emb = torch.index_select(self.node_emb, 0, h_nodes.view(-1)).view(batch, self.hist_len, -1)
        # torch.unsqueeze(input, dim, out=None) 返回在指定位置插入维度 size 为 1 的新张量.
        # s_node_emb.unsqueeze(1): b × 1 × d
        # h_node_emb: b × h × d
        # attention: b × h
        # attention = softmax((torch.mul(s_node_emb.unsqueeze(1), h_node_emb).sum(dim=2)), dim=1)

        # self.weight: d × d
        # self.bias: d
        # h_node_emb: b × h × d
        # hidden_h_node_emb: b × h × d
        hidden_h_node_emb = torch.relu(torch.matmul(h_node_emb, self.weight) + self.bias)
        # attention: b × h
        attention = softmax((torch.mul(s_node_emb.unsqueeze(1), hidden_h_node_emb).sum(dim=2)), dim=1)
        # b
        p_mu = torch.mul(s_node_emb, t_node_emb).sum(dim=1)
        # h_node_emb: b × h × d
        # t_node_emb.unsqueeze(1): b × 1 × d
        # p_alpha: b × h
        p_alpha = torch.mul(h_node_emb, t_node_emb.unsqueeze(1)).sum(dim=2)
        # @20201228增加softmax
        # p_alpha = softmax(torch.mul(h_node_emb, t_node_emb.unsqueeze(1)).sum(dim=2), dim=1)
        # if(self.epoch_temp == 9):
        #     print("p_alpha.data is: {}".format(p_alpha[0].data))
        self.temp_array1 += p_alpha.mean(dim=0).data.cpu().numpy()

        self.delta.data.clamp_(min=1e-6)
        self.beta.data.clamp_(min=1e-6)
        # delta: b × 1
        delta = torch.index_select(self.delta, 0, s_nodes.view(-1)).unsqueeze(1)
        beta = torch.index_select(self.beta, 0, s_nodes.view(-1)).unsqueeze(1)
        # t_times: b
        # t_times.unsqueeze(1): b × 1
        # h_times: b × h
        # d_time: b × h
        
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # (batch, hist_len)
        
        # ##########################################################
        ####使用haversine距离
        def haversine(lon_1, lat_1, lon_2, lat_2):
            """
            Calculate the great circle distance between two points
            on the earth (specified in decimal degrees)
            """
            # lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            # lat1_rad = torch.deg2rad(lat1.unsqueeze(1))
            # lon1_rad = torch.deg2rad(lon1.unsqueeze(1))
            # lat2_rad = torch.deg2rad(lat2)
            # lon2_rad = torch.deg2rad(lon2)
            # for j in range(lon_1.shape[0]):
            #     distances = []
            #     distance = []
            #     for i in range(lon_2.shape[1]):
            #         lon1, lat1, lon2, lat2 = map(radians, [lon_1[j], lat_1[j], lon_2[j,i], lat_2[j,i]])
            #         dlon = abs(lon2 - lon1)
            #         dlat = abs(lat2 - lat1)
            #         a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            #         c = 2 * asin(sqrt(a))
            #         r = 6371
            #         distance.append(c * r)
            #     distances.append(distance)
            # return distances
            def radians(degrees):
                return degrees * (math.pi / 180)
            # 使用 torch.Tensor.apply_() 将 radians 函数应用到每个元素
            lon_1.cpu().apply_(radians)
            lat_1.cpu().apply_(radians)
            lon_2.cpu().apply_(radians)
            lat_2.cpu().apply_(radians)
            
        
            dlon = torch.abs(lon_2 - lon_1)
            dlat = torch.abs(lat_2 - lat_1)
            a = torch.sin(dlat / 2) ** 2 + torch.cos(lat_1) * torch.cos(lat_2) * torch.sin(dlon / 2) ** 2
            c = 2 * torch.asin(torch.sqrt(a))
            r = 6371
            return c * r
        d_dict = haversine(t_loc_lon.unsqueeze(-1), t_loc_lat.unsqueeze(-1), h_locs_lon, h_locs_lat).to(self.device)
        #############
        # def calculate_distances(base_lat, base_lon, target_lat, target_lon): # [batch],[batch],[batch,3],[batch,3]
        #     # d_dict = []
        #     # 计算所有点之间的距离矩阵
        #     distances = vincenty_distance(base_lat, base_lon, target_lat, target_lon)

        #     # 计算每个点的面积
        #     areas = calculate_area(base_lat, base_lon, distances)

        #     # # 计算平均面积
        #     # distance_avg = torch.mean(areas, dim=1)

        #     return areas
        #         # d_dict.append(distances)
        #     # return d_dict


        # d_dict = torch.tensor(calculate_distances(t_loc_lat, t_loc_lon, h_locs_lat, h_locs_lon))
        # d_dict = torch.tensor(d_dict, device=self.device).clone().detach()
        ###使用欧氏距离
        # d_dict = torch.sqrt((t_loc_lat.unsqueeze(1) - h_locs_lat)**2 + (t_loc_lon.unsqueeze(1) - h_locs_lon)**2)
        #######################################################
        
        # 20201216防止d_time=0
        d_time.data.clamp_(min=1e-6)
        d_dict.data.clamp_(min=1e-6)
        if self.use_user_pref_attention:
            # s_node_emb.unsqueeze(1): b × 1 × d
            # h_node_emb: b × h × d
            # torch.cat([s_node_emb.unsqueeze(1), h_node_emb], dim=1): b × (h+1) × d
            # long_pref_weight = torch.sigmoid(torch.matmul(s_node_emb, self.long_pref_weight) + self.long_pref_bias)
            # short_pref_weight = torch.sigmoid(
            #     torch.matmul(torch.mean(h_node_emb, dim=1), self.short_pref_weight) + self.short_pref_bias)
            # long_pref_weight = torch.relu(torch.matmul(s_node_emb, self.long_pref_weight) + self.long_pref_bias)
            # short_pref_weight = torch.relu(
            #     torch.matmul(torch.mean(h_node_emb, dim=1), self.short_pref_weight) + self.short_pref_bias)
            # s_node_emb: b × d

            # 参考SHAN，将 u 替换为target user
            # long_pref_hidden 和 short_pref_hidden: b × d
            # t_node_emb：b × d
            # long_pref_hidden：b
            # long_pref_hidden = torch.sigmoid(torch.matmul(s_node_emb, self.long_pref_weight) + self.long_pref_bias)
            long_short_embedding = torch.cat([s_node_emb, torch.mean(h_node_emb, dim=1)], dim=1)
            # pref_hidden = torch.sigmoid(
            #     torch.matmul(long_short_embedding, self.long_short_pref_weight) + self.long_short_pref_bias)
            pref_hidden = torch.softmax(torch.relu(
                torch.matmul(long_short_embedding, self.long_short_pref_weight) + self.long_short_pref_bias), dim=1)
            # pref_weight = softmax(torch.cat([long_pref_hidden.unsqueeze(1), short_pref_hidden.unsqueeze(1)], dim=1),
            #                       dim=1)
            self.temp_value3 += pref_hidden[:, 0].mean().data
            self.temp_value4 += pref_hidden[:, 1].mean().data

            self.long_pref_weight = pref_hidden[:, 0]
            self.short_pref_weight = pref_hidden[:, 1]
            # long_pref_weight = (1.0 - pref_hidden)
            # short_pref_weight = pref_hidden
            # if(self.epoch_temp == 9):
            #     print("=============================================================")
            #     print("long_pref_hidden.data is: {}".format(long_pref_hidden[0].data))
            #     print("short_pref_hidden.data is: {}".format(short_pref_hidden[0].data))
            #     print("---------")
            #     print("long_pref_weight.data is: {}".format(long_pref_weight[0].data))
            #     print("short_pref_weight.data is: {}".format(short_pref_weight[0].data))
        else:
            self.long_pref_weight = torch.zeros(batch, dtype=torch.float) + 0.5
            self.short_pref_weight = torch.zeros(batch, dtype=torch.float) + 0.5
            if torch.cuda.is_available():
                self.long_pref_weight = self.long_pref_weight.to(self.device)
                self.short_pref_weight = self.short_pref_weight.to(self.device)
        self.temp_value1 += self.long_pref_weight.mean().data
        self.temp_value2 += self.short_pref_weight.mean().data

        if self.use_hist_attention:  # 对历史行为使用 attention机制
            # element-wise的乘法: *
            # torch.neg(delta) * d_time: b × h
            # Tensor与列向量做*乘法的结果是每行乘以列向量对应行的值（相当于把列向量的列复制，成为与lhs维度相同的Tensor）.
            # attention * p_alpha * torch.exp(torch.neg(delta) * d_time): b × h
            # p_lambda: b
            p_lambda = self.long_pref_weight * p_mu + self.short_pref_weight * (
                    attention * p_alpha * (torch.exp(torch.neg(delta) * d_time) * h_time_mask + torch.exp(torch.neg(beta) * d_dict) * h_time_mask)).sum(dim=1)

        else:
            p_lambda = self.long_pref_weight * p_mu + self.short_pref_weight * (
                    p_alpha * (torch.exp(torch.neg(delta) * d_time) * h_time_mask + torch.exp(torch.neg(beta) * d_dict) * h_time_mask)).sum(dim=1)
            # if (self.epoch_temp == 9):
            #     print("=============================================================")
            #     print("p_mu.data is: {}".format(p_mu[0].data))
            #     print("p_alpha.data is: {}".format(p_alpha[0].data))
            #     print("d_time.data is: {}".format(d_time[0].data))
            #     print("torch.exp(torch.neg(delta) * d_time).data is: {}".format(torch.exp(torch.neg(delta) * d_time)[0].data))
            #     print("(p_alpha * torch.exp(torch.neg(delta) * d_time) * h_time_mask).sum(dim=1).data is: {}".format((p_alpha * torch.exp(torch.neg(delta) * d_time) * h_time_mask).sum(dim=1)[0].data))

        # n_node_emb: b × neg_size × d
        n_node_emb = torch.index_select(self.node_emb, 0, n_nodes.view(-1)).view(batch, self.neg_size, -1)
        # n_mu: b × neg_size
        n_mu = torch.mul(s_node_emb.unsqueeze(1), n_node_emb).sum(dim=2)
        # h_node_emb.unsqueeze(2): b × h × 1 × d
        # n_node_emb.unsqueeze(1): b × 1 × neg_size × d
        # n_alpha: b × h × neg_size
        n_alpha = torch.mul(h_node_emb.unsqueeze(2), n_node_emb.unsqueeze(1)).sum(dim=3)
        # b × 1
        self.long_pref_weight = self.long_pref_weight.unsqueeze(1)
        self.short_pref_weight = self.short_pref_weight.unsqueeze(1)
        if self.use_hist_attention:
            # 概率密度：给定source节点和历史节点，计算负采样节点的条件概率密度
            # n_lambda: b × neg_size
            # n_lambda = n_mu + (attention.unsqueeze(2) * n_alpha * (torch.exp(torch.neg(delta) * d_time).unsqueeze(2)) * (
            #     h_time_mask.unsqueeze(2))).sum(dim=1)
            # n_lambda = long_pref_weight * n_mu + short_pref_weight * (
            #             attention.unsqueeze(2) * n_alpha * (torch.exp(torch.neg(delta) * d_time).unsqueeze(2)) * (
            #         h_time_mask.unsqueeze(2))).sum(dim=1)
            n_lambda = self.long_pref_weight.detach() * n_mu + self.short_pref_weight.detach() * (
                    attention.detach().unsqueeze(2) * n_alpha * ((torch.exp(torch.neg(delta) * d_time) + torch.exp(torch.neg(beta) * d_dict)).unsqueeze(2)) * (
                h_time_mask.unsqueeze(2))).sum(dim=1)
        else:
            # n_lambda = n_mu + (n_alpha * (torch.exp(torch.neg(delta) * d_time).unsqueeze(2)) * (
            #     h_time_mask.unsqueeze(2))).sum(dim=1)
            # n_lambda = long_pref_weight * n_mu + short_pref_weight * (
            #             n_alpha * (torch.exp(torch.neg(delta) * d_time).unsqueeze(2)) * (
            #         h_time_mask.unsqueeze(2))).sum(dim=1)
            n_lambda = self.long_pref_weight.detach() * n_mu + self.short_pref_weight.detach() * (
                    n_alpha * ((torch.exp(torch.neg(delta) * d_time) + torch.exp(torch.neg(beta) * d_dict) ).unsqueeze(2)) * (
                h_time_mask.unsqueeze(2))).sum(dim=1)

        self.temp_value5 += p_mu.mean().data
        self.temp_value6 += n_mu.mean().data
        return p_lambda, n_lambda

    def loss_func(self, s_nodes, t_nodes, t_times, t_loc_lat, t_loc_lon, n_nodes, h_nodes, h_times, h_locs_lat, h_locs_lon, h_time_mask):
        p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, t_loc_lat, t_loc_lon, n_nodes, h_nodes, h_times, h_locs_lat, h_locs_lon, h_time_mask)
        # loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) \
        #        - torch.log(torch.sigmoid(torch.neg(n_lambdas)) + 1e-6).sum(
        #     dim=1)/self.neg_size
        # bpr loss function
        loss = -torch.log(torch.sigmoid(p_lambdas.unsqueeze(1)-n_lambdas)).sum(dim=1)
        return loss

    def update(self, s_nodes, t_nodes, t_times, t_loc_lat, t_loc_lon, n_nodes, h_nodes, h_times, h_locs_lat, h_locs_lon, h_time_mask):
        self.opt.zero_grad()
        loss = self.loss_func(s_nodes, t_nodes, t_times, t_loc_lat, t_loc_lon, n_nodes, h_nodes, h_times, h_locs_lat, h_locs_lon, h_time_mask)
        loss = loss.sum()
        self.loss += loss.data
        loss.backward()
        self.opt.step()
        # print("loss :{}".format(loss))

    def train(self):
        self.epoch_temp = 0
        loss_sum = []
        for epoch in range(self.epochs):  # 循环训练
            self.epoch_temp = epoch
            self.temp_value1 = 0.0
            self.temp_value2 = 0.0
            self.temp_value3 = 0.0
            self.temp_value4 = 0.0
            self.temp_value5 = 0.0
            self.temp_value6 = 0.0
            self.temp_array1 = np.zeros(self.hist_len)
            self.loss = 0.0
            # 用DataLoader对数据集进行封装

            loader = DataLoader(self.data_tr, batch_size=self.batch, shuffle=True, num_workers=self.num_workers)
            # loader = DataLoader(self.data_tr, batch_size=self.batch, shuffle=True, num_workers=8)
            for i_batch, sample_batched in enumerate(loader):

                if torch.cuda.is_available():
                    self.update(sample_batched['source_node'].type(LType).to(self.device),
                                sample_batched['target_node'].type(LType).to(self.device),
                                sample_batched['target_time'].type(FType).to(self.device),
                                sample_batched['target_loc_lat'].type(FType).to(self.device),
                                sample_batched['target_loc_lon'].type(FType).to(self.device),
                                sample_batched['neg_nodes'].type(LType).to(self.device),
                                sample_batched['history_nodes'].type(LType).to(self.device),
                                sample_batched['history_times'].type(FType).to(self.device),
                                sample_batched['history_locs_lat'].type(FType).to(self.device),
                                sample_batched['history_locs_lon'].type(FType).to(self.device),
                                sample_batched['history_masks'].type(FType).to(self.device))
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['target_loc_lat'].type(FType),
                                sample_batched['target_loc_lon'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_locs_lat'].type(FType),
                                sample_batched['history_locs_lon'].type(FType),
                                sample_batched['history_masks'].type(FType))
            
            loss_sum.append(self.loss.cpu())

            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data_tr)) + '\n')
            sys.stdout.flush()
            # 每 test_and_save_step 次迭代之后就保存一次embedding并进行测试
            # if epoch % self.test_and_save_step == 0 and epoch != 0:
            # if ((epoch + 1) % self.test_and_save_step == 0) or epoch == 0:
            # if ((epoch + 1) % self.test_and_save_step == 0):
            if ((epoch + 1) % self.test_and_save_step == 0) or epoch == 0 or epoch == 4 or epoch == 9:
                # if epoch >= 0:
                # self.save_node_embeddings(
                #     './emb/lastfm_hhner_attn{}_{}_epochs{}_dim{}.emb'.format(self.use_attention, NORM_METHOD, epoch,
                #                                                              self.emb_size))
                # self.save_delta_value(
                #     './emb/lastfm_hhner_attn{}_{}_epochs{}.delta'.format(self.use_attention, NORM_METHOD, epoch))
                # 进行测试或者验证
                self.recommend(epoch, is_new_item=False)
                self.recommend(epoch, is_new_item=True)
            print("long_pref_weight.mean(): {}".format(self.temp_value1 / i_batch))
            print("short_pref_weight.mean(): {}".format(self.temp_value2 / i_batch))
            print("long_pref_hidden.mean(): {}".format(self.temp_value3 / i_batch))
            print("short_pref_hidden.mean(): {}".format(self.temp_value4 / i_batch))
            print("alpha.mean(): {}".format(self.temp_array1 / i_batch))
            print("p_mu.mean(): {}".format(self.temp_value5 / i_batch))
            print("n_mu.mean(): {}".format(self.temp_value6 / i_batch))
            print("==========================")
        
        # x = np.arange(0, len(loss_sum), 1) 
        # # 绘制
        # plt.figure()
        # plt.plot(x, loss_sum)

        # # 设置标题、坐标轴等
        # plt.title('Sample plot')  
        # plt.xlabel('x axis')
        # plt.ylabel('y axis')

        # # 显示 
        # # plt.show()
        # plt.savefig(str(top_n_freq) + " plot.png") 
        
        # self.save_node_embeddings('./emb/dblp_htne_attn_%d.emb' % (self.epochs))
        # self.save_parameter_value(
        #     './emb/NYC_hhner_attn{}_{}_epochs{}_dim{}_histlen{}.emb'.format(self.use_hist_attention, NORM_METHOD, self.epochs,
        #                                                              self.emb_size, self.hist_len), self.node_emb, "matrix")
        # self.save_parameter_value(
        #     './emb/NYC_hhner_attn{}_{}_epochs{}_dim{}_histlen{}.delta'.format(self.use_hist_attention, NORM_METHOD,
        #                                                                self.epochs,
        #                                                                self.emb_size, self.hist_len), self.delta, "vector")
        # self.save_parameter_value(
        #     './emb/NYC_hhner_attn{}_{}_epochs{}_dim{}_histlen{}.weight'.format(self.use_hist_attention, NORM_METHOD,
        #                                                                 self.epochs,
        #                                                                 self.emb_size, self.hist_len), self.weight, "matrix")
        # self.save_parameter_value(
        #     './emb/NYC_hhner_attn{}_{}_epochs{}_dim{}_histlen{}.bias'.format(self.use_hist_attention, NORM_METHOD, self.epochs,
        #                                                               self.emb_size, self.hist_len), self.bias, "vector")
        
        # self.save_parameter_value(
        #     './emb/NYC_hhner_attn{}_{}_epochs{}_dim{}_histlen{}.long_pref_weight'.format(self.use_hist_attention, NORM_METHOD,
        #                                                                           self.epochs,
        #                                                                           self.emb_size, self.hist_len), self.long_pref_weight,
        #     "matrix")
        # # self.save_parameter_value(
        # #     './emb/NYC_hhner_attn{}_{}_epochs{}_dim{}_histlen{}.long_pref_bias'.format(self.use_hist_attention, NORM_METHOD,
        # #                                                                         self.epochs,
        # #                                                                         self.emb_size, self.hist_len), self.long_pref_bias,
        # #     "vector")
        # self.save_parameter_value(
        #     './emb/NYC_hhner_attn{}_{}_epochs{}_dim{}_histlen{}.short_pref_weight'.format(self.use_hist_attention, NORM_METHOD,
        #                                                                            self.epochs,
        #                                                                            self.emb_size, self.hist_len),
        #     self.short_pref_weight, "matrix")
        # # self.save_parameter_value(
        # #     './emb/NYC_hhner_attn{}_{}_epochs{}_dim{}_histlen{}.short_pref_bias'.format(self.use_hist_attention, NORM_METHOD,
        # #                                                                          self.epochs,
        # #                                                                          self.emb_size, self.hist_len), self.short_pref_bias,"vector")

    def recommend(self, epoch, is_new_item=False):
        # hitrate/MRR
        count_all = 0
        rate_all_sum = 0
        recall_all_sum = np.zeros(self.top_n)
        MRR_all_sum = np.zeros(self.top_n)

        if is_new_item:  # next new item recommendation
            # windows 需要 num_workers=0
            loader = DataLoader(self.data_te_new, batch_size=self.batch, shuffle=False, num_workers=self.num_workers)
        else:  # next item recommendation
            # windows 需要 num_workers=0
            loader = DataLoader(self.data_te_old, batch_size=self.batch, shuffle=False, num_workers=self.num_workers)
        for i_batch, sample_batched in enumerate(loader):
            if torch.cuda.is_available():
                rate_all, recall_all, MRR_all = \
                    self.evaluate(sample_batched['source_node'].type(LType).to(self.device),
                                  sample_batched['target_node'].type(LType).to(self.device),
                                  sample_batched['target_time'].type(FType).to(self.device),
                                  sample_batched['target_loc_lat'].type(FType).to(self.device),
                                  sample_batched['target_loc_lon'].type(FType).to(self.device),
                                  sample_batched['history_nodes'].type(LType).to(self.device),
                                  sample_batched['history_times'].type(FType).to(self.device),
                                  sample_batched['history_locs_lat'].type(FType).to(self.device),
                                  sample_batched['history_locs_lon'].type(FType).to(self.device),
                                  sample_batched['history_masks'].type(FType).to(self.device))
            else:
                rate_all, recall_all, MRR_all = \
                    self.evaluate(sample_batched['source_node'].type(LType),
                                  sample_batched['target_node'].type(LType),
                                  sample_batched['target_time'].type(FType),
                                  sample_batched['target_loc_lat'].type(FType),
                                  sample_batched['target_loc_lon'].type(FType),
                                  sample_batched['history_nodes'].type(LType),
                                  sample_batched['history_times'].type(FType),
                                  sample_batched['history_locs_lat'].type(FType),
                                  sample_batched['history_locs_lon'].type(FType),
                                  sample_batched['history_masks'].type(FType))
            count_all += self.batch
            rate_all_sum += rate_all
            recall_all_sum += recall_all
            MRR_all_sum += MRR_all

        rate_all_sum_avg = rate_all_sum * 1. / count_all
        recall_all_avg = recall_all_sum * 1. / count_all
        MRR_all_avg = MRR_all_sum * 1. / count_all
        
        
       
                
        
        if is_new_item:
            for i in range(self.top_n):
                if recall_all_avg[i] > self.new_max_recall[i]:
                    self.new_max_recall[i] = recall_all_avg[i]
            for i in range(self.top_n):
                if MRR_all_avg[i] > self.new_max_mrr[i]:
                    self.new_max_mrr[i] = MRR_all_avg[i] 
            logging.info('=========== testing next new item epoch: {} ==========='.format(epoch))
            logging.info('count_all_next_new: {}'.format(count_all))
            logging.info('rate_all_sum_avg_next_new: {}'.format(rate_all_sum_avg))
            logging.info('recall_all_avg_next_new: {}'.format(recall_all_avg))
            logging.info('MRR_all_avg_next_new: {}'.format(MRR_all_avg))
            print('top_1: ', self.new_max_recall[0], 'top_5: ', self.new_max_recall[4], 'top_10: ', self.new_max_recall[9], 'top_20: ', self.new_max_recall[19])
            print('max_recall_top_n: ',self.new_max_recall)
            print('top_1: ', self.new_max_mrr[0], 'top_5: ', self.new_max_mrr[4], 'top_10: ', self.new_max_mrr[9], 'top_20: ', self.new_max_mrr[19])
            print('max_mrr_top_n: ',self.new_max_mrr)
        else:
            for i in range(self.top_n):
                if recall_all_avg[i] > self.max_recall[i]:
                    self.max_recall[i] = recall_all_avg[i]
                
            for i in range(self.top_n):
                if MRR_all_avg[i] > self.max_mrr[i]:
                    self.max_mrr[i] = MRR_all_avg[i]   
            logging.info('~~~~~~~~~~~~~ testing next item epoch: {} ~~~~~~~~~~~~~'.format(epoch))
            logging.info('count_all_next: {}'.format(count_all))
            logging.info('rate_all_sum_avg_next: {}'.format(rate_all_sum_avg))
            logging.info('recall_all_avg_next: {}'.format(recall_all_avg))
            logging.info('MRR_all_avg_next: {}'.format(MRR_all_avg))
            print('top_1: ', self.max_recall[0], 'top_5: ', self.max_recall[4], 'top_10: ', self.max_recall[9], 'top_20: ', self.max_recall[19])
            print('max_recall_top_n: ',self.max_recall)
            print('top_1: ', self.max_mrr[0], 'top_5: ', self.max_mrr[4], 'top_10: ', self.max_mrr[9], 'top_20: ', self.max_mrr[19])
            print('max_mrr_top_n: ',self.max_mrr)

    def evaluate(self, s_nodes, t_nodes, t_times, t_loc_lat, t_loc_lon, h_nodes, h_times, h_locs_lat, h_locs_lon, h_time_mask):
        batch = s_nodes.size()[0]
        # b × d
        # t_node_emb = torch.index_select(self.node_emb, 0, t_nodes.view(-1)).view(batch, -1)
        # self.node_emb: self.node_dim × self.emb_size
        # self.item_count × self.emb_size
        all_item_index = torch.arange(0, self.item_count)
        if torch.cuda.is_available():
            all_item_index = all_item_index.to(self.device)
        all_node_emb = torch.index_select(self.node_emb, 0, all_item_index).detach()

        # b × h × d
        # 返回的张量不与原始张量共享内存空间。
        h_node_emb = torch.index_select(self.node_emb, 0, h_nodes.view(-1)).detach().view(batch, self.hist_len, -1)
        # historical nodes influence target node
        p_alpha = torch.matmul(h_node_emb, torch.transpose(all_node_emb, 0, 1))
        # @20201228增加softmax
        # p_alpha = softmax(torch.matmul(h_node_emb, torch.transpose(all_node_emb, 0, 1)), dim=1)

        
        
        
        self.delta.data.clamp_(min=1e-6)
        self.beta.data.clamp_(min=1e-6)
        # d_time: b × h
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # (batch, hist_len)
        
        
        # ########################################################
        # ####全部的地点坐标都使用
        # def haversine(lon1, lat1, lon2, lat2):
        #     """
        #     Calculate the great circle distance between two points
        #     on the earth (specified in decimal degrees)
        #     """
        #     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        #     dlon = lon2 - lon1
        #     dlat = lat2 - lat1
        #     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        #     c = 2 * asin(sqrt(a))
        #     r = 6371
        #     return c * r
        # # d_dict = torch.sqrt((t_loc_lat.unsqueeze(1) - h_locs_lat)**2 + (t_loc_lon.unsqueeze(1) - h_locs_lon)**2)
        # def calculate_distances(base_lat, base_lon, target_lat, target_lon): # [batch],[batch],[batch,3],[batch,3]
        #     # d_dict = []
        #     # 计算所有点之间的距离矩阵
        #     # distances = vincenty_distance(base_lat, base_lon, target_lat, target_lon)
        #     lat1_rad = torch.deg2rad(base_lat.unsqueeze(-1))
        #     lon1_rad = torch.deg2rad(base_lon.unsqueeze(-1))
        #     lat2_rad = torch.deg2rad(target_lat.unsqueeze(0).unsqueeze(1).expand(batch, self.hist_len, self.item_count))
        #     lon2_rad = torch.deg2rad(target_lon.unsqueeze(0).unsqueeze(1).expand(batch, self.hist_len, self.item_count))

        #     dlon = lon2_rad - lon1_rad
        #     dlat = lat2_rad - lat1_rad

        #     a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
        #     c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        #     R = 6371.0  # 地球平均半径，单位为 km
            
        #     distance = R * c
        #     # 计算每个点的面积
        #     areas = calculate_area(base_lat, base_lon, distance)

        #     # # 计算平均面积
        #     # distance_avg = torch.mean(areas, dim=1)

        #     return areas

            
        # # d_dict = torch.tensor(calculate_distances(h_locs_lat, h_locs_lon, torch.tensor(self.poi_data[2].values).to(self.device), torch.tensor(self.poi_data[3].values).to(self.device))).sum(-1)
        # # d_dict = torch.tensor(d_dict, device=self.device).clone().detach()
        # # 计算历史节点的经纬度与所有节点的经纬度之间的差距
        # diff_lat = h_locs_lat.unsqueeze(-1) - torch.tensor(self.poi_data[2].values).unsqueeze(0).unsqueeze(1).expand(batch, self.hist_len, self.item_count).to(self.device)  # 扩展维度以便广播
        # diff_lon = h_locs_lon.unsqueeze(-1) - torch.tensor(self.poi_data[3].values).unsqueeze(0).unsqueeze(1).expand(batch, self.hist_len, self.item_count).to(self.device)  # 扩展维度以便广播

        # # 计算差距的欧几里德距离
        # d_dict = torch.sqrt(diff_lat ** 2 + diff_lon ** 2).sum(dim=-1)
        # # # 根据距离建模节点之间的影响
        # # # 这里仅作为示例，你可能需要根据具体情况设计影响度的计算方式
        # # d_dict = 1 / (1 + distance)  # 使用一个简单的函数，根据距离计算影响度
        
        ########从所有地点坐标中选择一个代表性的坐标
        ###使用haversine距离
        def haversine(lon_1, lat_1, lon_2, lat_2):
            """
            Calculate the great circle distance between two points
            on the earth (specified in decimal degrees)
            """
            # lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            # lat1_rad = torch.deg2rad(lat1.unsqueeze(-1))
            # lon1_rad = torch.deg2rad(lon1.unsqueeze(-1))
            # lat2_rad = torch.deg2rad(lat2)
            # lon2_rad = torch.deg2rad(lon2)

            # for j in range(lon_2.shape[0]):
            #     distances = []
            #     distance = []
            #     for i in range(lon_2.shape[1]):
            #         lon1, lat1, lon2, lat2 = map(radians, [lon_1, lat_1, lon_2[j,i], lat_2[j,i]])
            #         dlon = abs(lon2 - lon1)
            #         dlat = abs(lat2 - lat1)
            #         a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            #         c = 2 * asin(sqrt(a))
            #         r = 6371
            #         distance.append(c * r)
            #     distances.append(distance)
            # return distances
            
                # 定义 radians 函数
            def radians(degrees):
                return degrees * (math.pi / 180)
            # 使用 torch.Tensor.apply_() 将 radians 函数应用到每个元素
            lon_1.cpu().apply_(radians)
            lat_1.cpu().apply_(radians)
            lon_2.cpu().apply_(radians)
            lat_2.cpu().apply_(radians)
            
        
            dlon = torch.abs(lon_2 - lon_1)
            dlat = torch.abs(lat_2 - lat_1)
            a = torch.sin(dlat / 2) ** 2 + torch.cos(lat_1) * torch.cos(lat_2) * torch.sin(dlon / 2) ** 2
            c = 2 * torch.asin(torch.sqrt(a))
            r = 6371
            # dlon = torch.abs(lon2_rad - lon1_rad)
            # dlat = torch.abs(lat2_rad - lat1_rad)
            # a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1.unsqueeze(-1)) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
            # c = 2 * torch.asin(torch.sqrt(a))
            # r = 6371
            return c * r
        
        average_latitude = torch.tensor(np.mean(self.poi_data[2].values)).to(self.device)
        average_longitude = torch.tensor(np.mean(self.poi_data[3].values)).to(self.device)

        d_dict = haversine(average_longitude.unsqueeze(-1), average_latitude.unsqueeze(-1), h_locs_lon, h_locs_lat).to(self.device)
        
        # ###欧式距离
        # average_longitude = torch.tensor(np.mean(self.poi_data[2].values))
        # average_latitude = torch.tensor(np.mean(self.poi_data[3].values))
        # diff_lat = h_locs_lat - average_latitude
        # diff_lon = h_locs_lon - average_longitude
        # # 计算差距的欧几里德距离
        # d_dict = torch.sqrt(diff_lat ** 2 + diff_lon ** 2)
        # ################################################
        
        
        
        # delta: b × 1
        # 对于老用户来说是训练得到的值
        delta = torch.index_select(self.delta, 0, s_nodes.view(-1)).detach().unsqueeze(1)
        beta =  torch.index_select(self.beta, 0, s_nodes.view(-1)).detach().unsqueeze(1)
        # b × d
        s_node_emb = torch.index_select(self.node_emb, 0, s_nodes.view(-1)).detach().view(batch, -1)

        # s_node_emb.unsqueeze(1): b × 1 × d
        # h_node_emb: b × h × d
        # attention: b × h
        # attention = softmax((torch.mul(s_node_emb.unsqueeze(1), h_node_emb).sum(dim=2)), dim=1)

        # self.weight: d × d
        # self.bias: d
        # h_node_emb: b × h × d
        # hidden_h_node_emb: b × h × d
        hidden_h_node_emb = torch.relu(torch.matmul(h_node_emb, self.weight.detach()) + self.bias.detach())
        # attention: b × h
        attention = softmax((torch.mul(s_node_emb.unsqueeze(1), hidden_h_node_emb).sum(dim=2)), dim=1)
        # s_node_emb: b × d
        # self.node_emb: n × d
        # torch.transpose(self.node_emb, 0, 1): d × n
        # p_mu: b × n
        # source node (user) influence target node
        p_mu = torch.matmul(s_node_emb, torch.transpose(all_node_emb, 0, 1))
        if self.use_user_pref_attention:
            # long_pref_weight = torch.sigmoid(
            #     torch.matmul(s_node_emb, self.long_pref_weight.detach()) + self.long_pref_bias.detach())
            # short_pref_weight = torch.sigmoid(
            #     torch.matmul(torch.mean(h_node_emb, dim=1),
            #                  self.short_pref_weight.detach()) + self.short_pref_bias.detach())
            # long_pref_weight = torch.sigmoid(
            #     torch.matmul(s_node_emb, self.long_pref_weight.detach()) + self.long_pref_bias.detach())
            # short_pref_weight = torch.sigmoid(
            #     torch.matmul(torch.mean(h_node_emb, dim=1),
            #                  self.short_pref_weight.detach()) + self.short_pref_bias.detach())
            # long_pref_weight = torch.relu(
            #     torch.matmul(s_node_emb, self.long_pref_weight.detach()) + self.long_pref_bias.detach())
            # short_pref_weight = torch.relu(
            #     torch.matmul(torch.mean(h_node_emb, dim=1),
            #                  self.short_pref_weight.detach()) + self.short_pref_bias.detach())

            # long_pref_hidden 和 short_pref_hidden: b
            # long_pref_hidden = torch.relu(
            #     torch.matmul(s_node_emb, self.long_pref_weight.detach()) + self.long_pref_bias.detach())
            # short_pref_hidden = torch.relu(
            #     torch.matmul(torch.mean(h_node_emb, dim=1),
            #                  self.short_pref_weight.detach()) + self.short_pref_bias.detach())
            # pref_weight = softmax(torch.cat([long_pref_hidden.unsqueeze(1), short_pref_hidden.unsqueeze(1)], dim=1),
            #                       dim=1)
            # 参考SHAN，将 u 替换为target user
            # long_pref_hidden 和 short_pref_hidden: b × n
            # s_node_emb, t_node_emb：b × d
            # long_pref_weight: d × d
            # all_node_emb：n × d
            # long_pref_hidden：b x n

            long_short_embedding = torch.cat([s_node_emb, torch.mean(h_node_emb, dim=1)], dim=1)
            # pref_hidden = torch.sigmoid(
            #     torch.matmul(long_short_embedding, self.long_short_pref_weight.detach()) + self.long_short_pref_bias.detach())
            pref_hidden = torch.softmax(torch.relu(
                torch.matmul(long_short_embedding,
                             self.long_short_pref_weight.detach()) + self.long_short_pref_bias.detach()), dim=1)
            self.long_pref_weight = pref_hidden[:, 0]
            self.short_pref_weight = pref_hidden[:, 1]
        else:
            self.long_pref_weight = torch.zeros(batch, dtype=torch.float) + 0.5
            self.short_pref_weight = torch.zeros(batch, dtype=torch.float) + 0.5
            if torch.cuda.is_available():
                self.long_pref_weight = self.long_pref_weight.to(self.device)
                self.short_pref_weight = self.short_pref_weight.to(self.device)
        # b × 1
        self.long_pref_weight = self.long_pref_weight.unsqueeze(1)
        self.short_pref_weight = self.short_pref_weight.unsqueeze(1)
        if self.use_hist_attention:
            # p_mu: b × n
            # torch.exp(torch.neg(delta) * d_time): b × h
            # attention: b × h
            # p_alpha: b × h × n
            # h_time_mask: b × h
            # attention * torch.exp(torch.neg(delta) * d_time) * h_time_mask：
            # (attention * p_alpha * torch.exp(torch.neg(delta) * d_time) * h_time_mask).sum(dim=1):
            # p_lambda: b × n
            p_lambda = self.long_pref_weight * p_mu + self.short_pref_weight * (
                    p_alpha * (attention * (torch.exp(torch.neg(delta) * d_time) * h_time_mask + torch.exp(torch.neg(beta) * d_dict) * h_time_mask)).unsqueeze(2)).sum(
                dim=1)
        else:
            p_lambda = self.long_pref_weight * p_mu + self.short_pref_weight * (
                    p_alpha * ((torch.exp(torch.neg(delta) * d_time) * h_time_mask + torch.exp(torch.neg(beta) * d_dict) * h_time_mask)).unsqueeze(2)).sum(dim=1)

        rate_all_sum = 0
        recall_all = np.zeros(self.top_n)
        MRR_all = np.zeros(self.top_n)

        # p_lambda: b × n
        # 不推荐没有在训练集里出现过的item
        # for i in range(self.item_count):
        #     if i not in self.data_tr.item_node_set:
        #         p_lambda[:, i] = -sys.maxsize
        t_nodes_list = t_nodes.cpu().numpy().tolist()
        p_lambda_numpy = p_lambda.cpu().numpy()
        for i in range(len(t_nodes_list)):
            t_node = t_nodes_list[i]
            p_lambda_numpy_i_item = p_lambda_numpy[i]  # 第i个batch，所有item（不包括用户）
            # 降序排序
            prob_index = np.argsort(-p_lambda_numpy_i_item).tolist()
            gnd_rate = prob_index.index(t_node) + 1
            rate_all_sum += gnd_rate
            if gnd_rate <= self.top_n:
                recall_all[gnd_rate - 1:] += 1
                MRR_all[gnd_rate - 1:] += 1. / gnd_rate
        return rate_all_sum, recall_all, MRR_all

    def save_parameter_value(self, path, parameter, data_type):
        if torch.cuda.is_available():
            parameter_cpu = parameter.cpu().data.numpy()
        else:
            parameter_cpu = parameter.data.numpy()
        writer = open(path, 'w')
        if data_type == "vector":
            writer.write('%d\n' % (parameter_cpu.shape[0]))
            writer.write('\t'.join(str(d) for d in parameter_cpu))
        elif data_type == "matrix":
            dim_0, dim_1 = parameter_cpu.shape
            writer.write('%d\t%d\n' % (dim_0, dim_1))
            for n_idx in range(dim_0):
                writer.write('\t'.join(str(d) for d in parameter_cpu[n_idx]) + '\n')
        else:
            pass
        writer.close()