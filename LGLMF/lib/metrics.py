__author__ = 'YidingLiu'

import numpy as np


def mapk(actual, predicted, k):
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def precisionk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(predicted)


def recallk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(actual)


def ndcgk(actual, predicted):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i,p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    return dcg / idcg

def mrrk(actual, predicted):
    mrr_sum = 0.0
    for act in actual:
        if act in predicted:
            rank = predicted.index(act) + 1  # 找到 act 在 predicted 中的位置（排名）
            mrr_sum += 1.0 / rank  # 计算当前 act 的倒数排名并累加到总和
    return mrr_sum / len(actual)  # 计算平均倒数排名