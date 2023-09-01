import numpy as np
 
def get_metric(t_nodes,p_lambda,top_n): # 真实值，预测值
    rate_all_sum = 0
    recall_all = np.zeros(top_n)
    MRR_all = np.zeros(top_n)

    # p_lambda: b × n
    # 不推荐没有在训练集里出现过的item
    # for i in range(self.item_count):
    #     if i not in self.data_tr.item_node_set:
    #         p_lambda[:, i] = -sys.maxsize
    t_nodes_list = t_nodes.cpu().numpy().tolist()
    p_lambda_numpy = p_lambda.cpu().detach().numpy()
    for i in range(len(t_nodes_list)):
        t_node = t_nodes_list[i]
        p_lambda_numpy_i_item = p_lambda_numpy[i]  # 第i个batch，所有item（不包括用户）
        # 降序排序
        prob_index = np.argsort(-p_lambda_numpy_i_item).tolist()
        gnd_rate = prob_index.index(t_node) + 1
        rate_all_sum += gnd_rate
        if gnd_rate <= top_n:
            recall_all[gnd_rate - 1:] += 1
            MRR_all[gnd_rate - 1:] += 1. / gnd_rate
    return rate_all_sum, recall_all, MRR_all


import math
import torch

def vincenty_distance(lat1, lon1, lat2, lon2):
    lat1_rad = torch.deg2rad(lat1.unsqueeze(1))
    lon1_rad = torch.deg2rad(lon1.unsqueeze(1))
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    R = 6371.0  # 地球平均半径，单位为 km
    distance = R * c
    return distance

# # 示例：计算两点距离
# lat1 = 37.7749  # 纬度1
# lon1 = -122.4194  # 经度1
# lat2 = 34.0522  # 纬度2
# lon2 = -118.2437  # 经度2

# distance = vincenty_distance(lat1, lon1, lat2, lon2)
# print("距离：", distance, "米")


import math

# 地球的赤道周长（单位：公里）
equatorial_circumference = 40075.0

def calculate_area(latitude, longitude, s):
    # 将长度s转换为角度
    angle = (s / equatorial_circumference) * 360.0

    # 计算纬度范围
    lat_range = angle / torch.cos(torch.deg2rad(latitude)).unsqueeze(-1)

    # 计算以给定纬度为圆心的经度范围
    lon_range = angle /  torch.cos(torch.deg2rad(longitude)).unsqueeze(-1)

    # 计算面积（使用简化的平面几何方法，可能不精确）
    area = lat_range * lon_range * (math.pi / 180.0) * (equatorial_circumference ** 2) / 4.0

    return area

# # 示例经纬度和长度
# latitude = 37.7749  # 旧金山的纬度
# longitude = -122.4194  # 旧金山的经度
# s = 100.0  # 长度为100公里

# # 计算面积
# result = calculate_area(latitude, longitude, s)

# print(f"以 ({latitude}, {longitude}) 为圆心，长度 {s} 公里的圆形区域的面积为 {result} 平方公里")

       