# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import numpy as np
import sys
import random


class DataSetTrain(Dataset):

    def __init__(self, train_path, user_count=0, item_count=0, neg_size=5, hist_len=2, directed=False):
        '''

        :param train_path:
        :param user_count: 用户数目
        :param item_coun：物品数目
        :param neg_size: 负采样的个数
        :param hist_len:
        :param directed:
        :param transform:
        '''
        self.neg_size = neg_size
        self.user_count = user_count
        self.item_count = item_count
        self.hist_len = hist_len
        self.directed = directed
        # self.transform = transform

        # AttributeError: module 'sys' has no attribute 'maxint'
        # self.max_d_time = -sys.maxint  # Time interval [0, T]
        # self.max_d_time = -sys.maxsize  # Time interval [0, T]，但是这个是负的？配合后面的代码统计最大的时间戳

        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)  # 如果节点多，感觉可以适当增加

        # 节点及与其相邻的其它节点
        self.node2hist = dict()  # source_node_1---[target_node_1, target_node_2, ......, targetnode_n]
        # self.node_set = set()
        self.user_item_dict = dict()
        self.user_node_set = set()
        self.item_node_set = set()
        self.degrees = dict()  # 出度+入度
        print(train_path)
        with open(train_path, 'r') as infile:
            for line in infile:
                # print(line)
                parts = line.split()
                s_node = int(parts[0])  # source node
                t_node = int(parts[1])  # target node
                time_stamp = float(parts[2])  # time slot, delta t
                t_loc_x = float(parts[3])
                t_loc_y = float(parts[4])
                # update将元素添加到set，可以添加多个元素
                # self.node_set.update([s_node, t_node])
                if s_node not in self.user_item_dict:
                    self.user_item_dict[s_node] = set()
                self.user_item_dict[s_node].add(t_node)

                self.user_node_set.add(s_node)
                self.item_node_set.add(t_node)

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, time_stamp, t_loc_x, t_loc_y))

                if not directed:  # 非有向图，一条边的数据需要添加两次
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    self.node2hist[t_node].append((s_node, time_stamp, t_loc_x, t_loc_y))

                # if d_time > self.max_d_time:
                #     self.max_d_time = d_time

                if s_node not in self.degrees:
                    self.degrees[s_node] = 0
                if t_node not in self.degrees:
                    self.degrees[t_node] = 0
                self.degrees[s_node] += 1
                self.degrees[t_node] += 1

        # self.node_dim = len(self.node_set)  # 所有节点数目
        # 有些节点被过滤掉了，因此根据数据集处理过程，指定node个数
        self.node_dim = self.user_count + self.item_count  # 所有节点数目

        self.data_size = 0  # 所有边的数目
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])  # 历史节点按照时间排序
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        '''
        node2hist：边，0-1，0-2；1-0，1-2；2-1
        {0: [(1, 1.0), (2, 1.0)], 1: [(0, 1.0), (2, 1.0)], 2: [(1, 1.0)]}
        idx2source_id：0作为source出现了2次，1作为source出现了2次，2作为source出现了1次
        [0, 0, 1, 1, 2]
        idx2target_id：node2hist中的target的位置，例如0-1中的1出现在node2hist[0]的位置0处
        [0, 1, 0, 1, 0]
        '''
        # 存储source_node 的id
        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        # 存储target_node在对应source_node的 node2hist 里的 index（位置）
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0  # 所有边的索引，从0到self.data_size-1
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                idx += 1

        self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table()

    def get_node_dim(self):
        '''
        所有节点个数
        :return:
        '''
        return self.node_dim

    # def get_max_d_time(self):
    #     return self.max_d_time

    def init_neg_table(self):
        total_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        for k in range(self.node_dim):  # 计算所有节点的度之和
            # total_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)
            if k in self.degrees:  # 有些节点不存在
                total_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)
            else:
                self.degrees[k] = 0
        # for k in xrange(self.neg_table_size):
        for k in range(self.neg_table_size):  # 构建neg_table
            if (k + 1.) / self.neg_table_size > por:
                while self.degrees[n_id] == 0:
                    n_id += 1
                cur_sum += np.power(self.degrees[n_id], self.NEG_SAMPLING_POWER)
                por = cur_sum / total_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]
        t_loc_x = self.node2hist[s_node][t_idx][2]
        t_loc_y = self.node2hist[s_node][t_idx][3]
        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]
        hist_loc_x = [h[2] for h in hist]
        hist_loc_y = [h[3] for h in hist]

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_locs_x = np.zeros((self.hist_len, ))
        np_h_locs_x[:len(hist_loc_x)] = hist_loc_x
        np_h_locs_y = np.zeros((self.hist_len, ))
        np_h_locs_y[:len(hist_loc_y)] = hist_loc_y
        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = 1.

        # neg_nodes = self.negative_sampling()
        neg_nodes = self.negative_sampling(s_node, t_node)

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'target_loc_lat': t_loc_x,
            'target_loc_lon': t_loc_y,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_locs_lat': np_h_locs_x,
            'history_locs_lon': np_h_locs_y,
            'history_masks': np_h_masks,
            'neg_nodes': neg_nodes,
        }

        # if self.transform:
        #     sample = self.transform(sample)

        return sample

    # def negative_sampling(self):
    #     rand_idx = np.random.randint(0, self.neg_table_size, (self.neg_size,))
    #     sampled_nodes = self.neg_table[rand_idx]
    #     return sampled_nodes
    # 避免 negative nodes与 source_nodes和 target_node一样
    # 我们用的是有向图，user-item，因此 negative_nodes 需要是音乐节点
    def negative_sampling(self, source_node, target_node):
        sampled_nodes = []
        func = lambda: self.neg_table[np.random.randint(0, self.neg_table_size)]
        for i in range(self.neg_size):
            temp_neg_node = func()
            while temp_neg_node == source_node or temp_neg_node == target_node or temp_neg_node >= self.item_count:
                temp_neg_node = func()
            # sample_edges.append([node1[i], node2[i], neg_node[0], neg_node[1], weight[i]])
            sampled_nodes.append(temp_neg_node)
        return np.array(sampled_nodes)


class DataSetTestNext(Dataset):

    def __init__(self, file_path, user_count=0, item_count=0, hist_len=2, user_item_dict=None, directed=False):
        '''
        :param file_path:
        :param user_count: 用户数目
        :param item_count：物品数目
        :param hist_len:
        :param node_set_in_train: 训练集中出现过的用户和物品节点
        :param user_item_dict:
        :param directed:
        '''
        self.user_count = user_count
        self.item_count = item_count
        self.hist_len = hist_len
        # self.node_set_in_train = node_set_in_train
        self.directed = directed
        # self.transform = transform

        # 节点及与其相邻的其它节点
        self.node2hist = dict()  # source_node_1---[target_node_1, target_node_2, ......, targetnode_n]
        # self.node_set_not_in_train = set()
        with open(file_path, 'r') as infile:
            for line in infile:
                parts = line.split()
                s_node = int(parts[0])  # source node
                t_node = int(parts[1])  # target node
                time_stamp = float(parts[2])  # time slot, delta t
                t_loc_x = float(parts[3])
                t_loc_y = float(parts[4])
                # 保证测试集的数据在训练集出现了
                # s_node（old_user ） 肯定在 self.node_set_in_train 里面
                # if t_node not in self.node_set_in_train:
                #     self.node_set_not_in_train.add(t_node)
                #     continue

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, time_stamp, t_loc_x, t_loc_y))

                if not directed:  # 非有向图，一条边的数据需要添加两次
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    self.node2hist[t_node].append((s_node, time_stamp, t_loc_x, t_loc_y))

        self.data_size = 0  # 所有边的数目
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])  # 历史节点按照时间排序
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        # 存储source_node 的id
        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        # 存储target_node在对应source_node的 node2hist 里的 index（位置）
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0  # 所有边的索引，从0到self.data_size-1
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                idx += 1

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]
        t_loc_x = self.node2hist[s_node][t_idx][2]
        t_loc_y = self.node2hist[s_node][t_idx][3]
        
        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]
        hist_loc_x = [h[2] for h in hist]
        hist_loc_y = [h[3] for h in hist]

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_locs_x = np.zeros((self.hist_len, ))
        np_h_locs_x[:len(hist_loc_x)] = hist_loc_x
        np_h_locs_y = np.zeros((self.hist_len, ))
        np_h_locs_y[:len(hist_loc_y)] = hist_loc_y
        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = 1.

        # neg_nodes = self.negative_sampling()
        # neg_nodes = self.negative_sampling(s_node, t_node)

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'target_loc_lat': t_loc_x,
            'target_loc_lon': t_loc_y,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_locs_lat': np_h_locs_x,
            'history_locs_lon': np_h_locs_y,
            'history_masks': np_h_masks,
        }

        return sample


class DataSetTestNextNew(Dataset):

    def __init__(self, file_path, user_count=0, item_count=0, hist_len=2, user_item_dict=None, directed=False):
        '''
        :param file_path:
        :param user_count: 用户数目
        :param item_count：物品数目
        :param hist_len:
        :param node_set_in_train: 训练集中出现过的用户和物品节点
        :param user_item_dict: user1->(item1, item2)......
        :param directed:
        '''
        self.user_count = user_count
        self.item_count = item_count
        self.hist_len = hist_len
        # self.node_set_in_train = node_set_in_train
        self.user_item_dict = user_item_dict
        self.directed = directed
        # self.transform = transform

        # 节点及与其相邻的其它节点
        self.node2hist = dict()  # source_node_1---[target_node_1, target_node_2, ......, targetnode_n]
        # self.node_set_not_in_train = set()
        with open(file_path, 'r') as infile:
            for line in infile:
                parts = line.split()
                s_node = int(parts[0])  # source node
                t_node = int(parts[1])  # target node
                time_stamp = float(parts[2])  # time slot, delta t
                t_loc_x = float(parts[3])
                t_loc_y = float(parts[4])
                # 保证测试集的数据在训练集出现了
                # s_node（old_user ） 肯定在 self.node_set_in_train 里面
                # if t_node not in self.node_set_in_train:
                #     self.node_set_not_in_train.add(t_node)
                #     continue

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, time_stamp, t_loc_x, t_loc_y))

                if not directed:  # 非有向图，一条边的数据需要添加两次
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    self.node2hist[t_node].append((s_node, time_stamp, t_loc_x, t_loc_y))

        self.data_size = 0  # 所有边的数目
        # for s in self.node2hist:
        #     hist = self.node2hist[s]
        #     hist = sorted(hist, key=lambda x: x[1])  # 历史节点按照时间排序
        #     self.node2hist[s] = hist
        #     self.data_size += len(self.node2hist[s])
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])  # 历史节点按照时间排序
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        # 存储source_node 的id
        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        # 存储target_node在对应source_node的 node2hist 里的 index（位置）
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0  # 所有边的索引，从0到self.data_size-1
        for s_node in self.node2hist:
            s_node_hist = self.node2hist[s_node]
            # s_node_current_hist_set = set()
            s_node_current_hist_set = set(self.user_item_dict[s_node])
            for t_idx in range(len(s_node_hist)):
                # target item需要是new item
                if s_node_hist[t_idx][0] not in s_node_current_hist_set:
                    self.idx2source_id[idx] = s_node
                    self.idx2target_id[idx] = t_idx
                    idx += 1
                    s_node_current_hist_set.add(s_node_hist[t_idx][0])
        self.data_size = idx
        # 存储source_node 的id
        self.idx2source_id = self.idx2source_id[:self.data_size]
        # 存储target_node在对应source_node的 node2hist 里的 index（位置）
        self.idx2target_id = self.idx2target_id[:self.data_size]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]
        t_loc_x = self.node2hist[s_node][t_idx][2]
        t_loc_y = self.node2hist[s_node][t_idx][3]
        #
        # while True:
        #
        #     break

        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]
        hist_loc_x = [h[2] for h in hist]
        hist_loc_y = [h[3] for h in hist]

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_locs_x = np.zeros((self.hist_len, ))
        np_h_locs_x[:len(hist_loc_x)] = hist_loc_x
        np_h_locs_y = np.zeros((self.hist_len, ))
        np_h_locs_y[:len(hist_loc_y)] = hist_loc_y
        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = 1.

        # neg_nodes = self.negative_sampling()
        # neg_nodes = self.negative_sampling(s_node, t_node)

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'target_loc_lat': t_loc_x,
            'target_loc_lon': t_loc_y,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_locs_lat': np_h_locs_x,
            'history_locs_lon': np_h_locs_y,
            'history_masks': np_h_masks,
        }

        return sample
