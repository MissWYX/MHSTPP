import argparse
import numpy as np
import pickle
import time
import metric
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import os.path


from preprocess.Dataset import get_dataloader
if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T

import optuna

# import transformer.Constants as Constants
import Utils

from preprocess.DatasetTrajectory import Dataset as dataset
from transformer.Models import Transformer
from tqdm import tqdm

top_n =30

def vaild(prediction, label, top_n, pre, rec, map_, ndcg, mrr):
    top_ = torch.topk(prediction, top_n, -1, sorted=True)[1]  # (32, 30)
    for top, l in zip(top_, label):
        l = l[l != 0]
        l = l - 1
        recom_list, ground_list = top.cpu().numpy(), l.cpu().numpy()
        if len(ground_list) == 0:
            continue
        # map2, mrr, ndcg2 = metric.map_mrr_ndcg(recom_list, ground_list)
        pre2, rec2, map2, ndcg2, mrr2 = metric.precision_recall_ndcg_at_k(top_n, recom_list, ground_list)
        pre.append(pre2)
        rec.append(rec2)
        map_.append(map2)
        ndcg.append(ndcg2)
        mrr.append(mrr2)


def pre_rec_top(pre, rec, map_, ndcg, mrr, prediction, label, target_):
    prediction = prediction * target_
    for i, topN in enumerate(np.arange(1, 31)):
        vaild(prediction, label, topN, pre[i], rec[i], map_[i], ndcg[i], mrr[i])


def train_epoch(model,  place_correlation, user_dl, poi_dl, pred_loss_func, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    pre = [[] for i in range(30)]
    rec = [[] for i in range(30)]
    map_ = [[] for i in range(30)]
    ndcg = [[] for i in range(30)]
    mrr = [[] for i in range(30)]
    for batch in tqdm(user_dl, mininterval=2, desc='  - (Training)   ', leave=False):
        # training user
        """ prepare user data """
        event_type, test_label, inner_dis, user_type = map(lambda x: x.to(opt.device), batch)
        # event_type, score, test_label, test_score, inner_dis = map(lambda x: x.to(opt.device), batch)
        """ forward """
        optimizer.zero_grad()
        type_prediction, target_ = model(event_type, inner_dis, user_type, True)  # X = (UY+Z) ^ T
        # enc_output, user_type = model(event_type, inner_dis, user_type, True)  # X = (UY+Z) ^ T
        # enc_output = model(event_type, None, None, user_type, False)  # X = (UY+Z) ^ T
        # type_prediction, target_ = task1(enc_output, event_type, user_type)  # [16, 105, 22]
        prediction = torch.squeeze(type_prediction, 1)
        pre_rec_top(pre, rec, map_, ndcg, mrr, prediction, test_label, target_)
        """ backward """
        loss = Utils.rating_loss(prediction, event_type, test_label, pred_loss_func, top_n_freq)

        loss.backward(retain_graph=True)
        """ update parameters """
        optimizer.step()

    pre_np, rec_np, map_np, ndcg_np, mrr_np = np.zeros(30), np.zeros(30), np.zeros(30), np.zeros(30), np.zeros(30)
    for i in range(30):
        pre_np[i], rec_np[i], map_np[i], ndcg_np[i], mrr_np[i]= np.mean(pre[i]), np.mean(rec[i]), np.mean(map_[i]), np.mean(ndcg[i]), np.mean(mrr[i])

    return pre_np, rec_np, map_np, ndcg_np, mrr_np


def eval_epoch(model,  place_correlation, user_valid_dl, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    pre = [[] for i in range(30)]
    rec = [[] for i in range(30)]
    map_ = [[] for i in range(30)]
    ndcg = [[] for i in range(30)]
    mrr = [[] for i in range(30)]


    with torch.no_grad():
        for batch in tqdm(user_valid_dl, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_type, test_label, inner_dis, user_type = map(lambda x: x.to(opt.device), batch)
            # event_type, score, test_label, test_score, inner_dis = map(lambda x: x.to(opt.device), batch)
            """ forward """

            type_prediction, target_ = model(event_type, inner_dis, user_type, True)  # X = (UY+Z) ^ T
            # enc_output = model(event_type, None, None, user_type, False)  # X = (UY+Z) ^ T
            # type_prediction, target_ = task1(enc_output, event_type, model.encoder.event_emb, place_correlation)  # [16, 105, 22]
            # type_prediction, target_ = task1(enc_output, event_type, user_type)  # [16, 105, 22]
            prediction = torch.squeeze(type_prediction, 1)
            pre_rec_top(pre, rec, map_, ndcg, mrr, prediction, test_label, target_)

    pre_np, rec_np, map_np, ndcg_np, mrr_np = np.zeros(30), np.zeros(30), np.zeros(30), np.zeros(30), np.zeros(30)
    for i in range(30):
        pre_np[i], rec_np[i], map_np[i], ndcg_np[i], mrr_np[i]= np.mean(pre[i]), np.mean(rec[i]), np.mean(map_[i]), np.mean(ndcg[i]), np.mean(mrr[i])

    return pre_np, rec_np, map_np, ndcg_np, mrr_np




def train(model, data, pred_loss_func, optimizer, scheduler, opt):
    """ Start training. """

    valid_precision_max = 0.0
    (place_correlation, poi_dl, user_valid_dl, user_dl) = data
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')
        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
        start = time.time()  # loglikelihood: {ll: 8.5f},
        pre_np, rec_np, map_np, ndcg_np, mrr_np = train_epoch(model, place_correlation, user_dl, poi_dl, pred_loss_func, optimizer, opt)
        print('\r (Training)P@k:{pre}, R@k:{rec}, \n'
              '(Training)map@k:{map_}, ndcg@k:{ndcg}, mrr@k:{mrr},'
              'elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60,
                      pre=pre_np, rec=rec_np, map_=map_np, ndcg=ndcg_np, mrr=mrr_np))

        start = time.time()
        pre_np, rec_np, map_np, ndcg_np, mrr_np = eval_epoch(model,  place_correlation, user_valid_dl, opt)
    
        
        print('\r (Test)P@k:{pre}, R@k:{rec}, \n'
              '(Test)map@k:{map_}, ndcg@k:{ndcg}, mrr@k:{mrr},'
              'elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60,
                      pre=pre_np, rec=rec_np, map_=map_np, ndcg=ndcg_np, mrr=mrr_np))
        scheduler.step()

        valid_precision_max = valid_precision_max if valid_precision_max > pre_np[0] else pre_np[0]
    return valid_precision_max


def main(trial, dataset_name, user_num, top_n_item):
    """ Main function. """
    parser = argparse.ArgumentParser()
    # parser.add_argument('-data', required=True)
    #
    # parser.add_argument('-epoch', type=int, default=30)
    # parser.add_argument('-batch_size', type=int, default=16)
    #
    # parser.add_argument('-d_model', type=int, default=64)
    # parser.add_argument('-d_rnn', type=int, default=256)
    # parser.add_argument('-d_inner_hid', type=int, default=128)
    # parser.add_argument('-d_k', type=int, default=16)
    # parser.add_argument('-d_v', type=int, default=16)
    #
    # parser.add_argument('-n_head', type=int, default=4)
    # parser.add_argument('-n_dis', type=int, default=4)
    # parser.add_argument('-n_layers', type=int, default=4)
    #
    # parser.add_argument('-dropout', type=float, default=0.1)
    # parser.add_argument('-lr', type=float, default=1e-4)
    # parser.add_argument('-smooth', type=float, default=0.1)
    #
    # parser.add_argument('-log', type=str, default='log.txt')
    #
    # parser.add_argument('-ita', type=float, default=0.05)
    #
    # parser.add_argument('-coefficient', type=float, default=0.05)

    opt = parser.parse_args()

    import sys
    # print("Python Version {}".format(str(sys.version).replace('\n', '')))
    # print(torch.cuda.is_available())
    # default device is CUDA
    opt.device = torch.device('cuda:1')

    # # setup the log file
    # with open(opt.log, 'w') as f:
    #     f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    # """ prepare dataloader """
    # trainloader, testloader = prepare_dataloader(opt)
    # trainloader, testloader, num_types = prepare_dataloader(opt)
    # trainloader, testloader = [], []

    """ prepare model """
    # opt.n_layers = trial.suggest_int('n_layers', 2, 2)
    # opt.d_inner_hid = trial.suggest_int('n_hidden', 512, 1024, 128)
    # opt.d_k = trial.suggest_int('d_k', 512, 1024, 128)
    # opt.d_v = trial.suggest_int('d_v', 512, 1024, 128)
    # opt.n_head = trial.suggest_int('n_head', 8, 12, 2)
    # opt.n_dis = trial.suggest_int('n_dis', 8, 12, 2)
    # # opt.d_rnn = trial.suggest_int('d_rnn', 128, 512, 128)
    # opt.d_model = trial.suggest_int('d_model', 1152, 1280, 128)
    # opt.dropout = trial.suggest_uniform('dropout_rate', 0.5, 0.7)
    # opt.smooth = trial.suggest_uniform('smooth', 1e-2, 1e-1)
    # opt.lr = trial.suggest_uniform('learning_rate', 0.00008, 0.00011)
    #
    # opt.ita = trial.suggest_uniform('ita', 0.03, 0.06)
    # opt.coefficient = trial.suggest_uniform('coefficient', 0.05, 0.15)

    opt.lr = 0.0001
    opt.batch_size = 32
    opt.epoch = 30
    # # #
    opt.n_layers = 2  # 2
    opt.d_inner_hid = 256  # 768
    opt.d_rnn = 128
    opt.d_model = 1024
    opt.d_k = 256
    opt.d_v = 256
    opt.n_head = 4  # 8
    opt.n_dis = 4
    opt.dropout = 0.5
    opt.smooth = 0.1
    opt.ita = 0.037
    opt.coefficient = 0.14

    print('[Info] parameters: {}'.format(opt))
    # model = torch.load('./model/STaTRL.pth.tar')
    # model = torch.load('./{}_model/best.pth.tar'.format(Constants.DATASET))
    # num_types = Constants.TYPE_NUMBER
    num_types = top_n_item
    # num_user = Constants.USER_NUMBER
    model = Transformer(
        num_types= num_types,
        # user_num = user_num,
        d_model=opt.d_model,
        # disc=disc,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        batch_size=opt.batch_size,
        device=opt.device,
        ita=opt.ita,
        n_dis=opt.n_dis
    )
    model.to(opt.device)

    """ loading data"""

    ds = dataset(dataset_name, user_num, top_n_item)
    place_correlation = ds.place_coords
    poi_dl = []  # ds.get_poi_dl(opt.batch_size)
    print('[Info] Loading test data...')
    user_valid_dl = ds.get_user_valid_dl(opt.batch_size)
    print('[Info] Loading training data...')
    user_dl = ds.get_user_dl(opt.batch_size)

    data = (place_correlation, poi_dl, user_valid_dl, user_dl)

    """ optimizer and scheduler """
    parameters = [
                  {'params': model.parameters(), 'lr': opt.lr},
                  ]
    optimizer = torch.optim.Adam(parameters)  # , weight_decay=0.01
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, opt.device, opt.coefficient, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    return train(model, data, pred_loss_func, optimizer, scheduler, opt)

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # main()

    # df = study.trials_dataframe()
    #
    # print("Best trial:")
    # trial = study.best_trial
    # print("  Value: ", trial.value)
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
    
            # 文件格式： 节点1，节点2，时间
    # last fm，user=992，music=5000

    NORM_METHOD = 'hour'
    print('NORM_METHOD: {}'.format(NORM_METHOD))

    # data_index = 2  # gowalla
    dataset_name = "gowalla"

    min_length = 100 # more than
    max_length = 500  # less than

    # 30_user = config.user_count_list[data_index]
    user_num = 2000
    PAD = 0
    for top_n_freq in [10000, 15000, 20000, 25000, 30000]:
    # for top_n_freq in [9000, 8000, 7000, 6000, 5000]:    
    # for 30_freq in config.30_item_list:
        # logging.info('p_lambda = long_pref_weight * p_mu + short_pref_weight * (p_alpha * (torch.exp(torch.neg(delta) * d_time) * h_time_mask + torch.exp(torch.neg(beta) * d_dict) * h_time_mask)).sum(dim=1)')
        logging.info('+++++++++++++ start top_n_item is: {} ++++++++++++++++'.format(top_n_freq))
        logging.info('+++++++++++++ start top_n_item is: {} ++++++++++++++++'.format(top_n_freq))
        logging.info('+++++++++++++ start top_n_item is: {} ++++++++++++++++'.format(top_n_freq))
        # BASE_DIR = ''
        # DATA_SOURCE = 'gowalla'
        # path = os.path.join(BASE_DIR, DATA_SOURCE)
        # train_path = path + '/SAE_NAD_tr_top{}.lst'.format(30_freq)
        # # test_old_path = path +  '/SAE_NAD_tr_top{}.lst'.format(30_freq)
        # poi_data = pd.read_csv(path+'/poi_top_'+str(30_freq)+'.csv', header=None)
        # test_new_path = './{}/te-user-new-item-time-top{}-min{}-max{}-{}.lst'.format(
        #     dataset, 30_freq, min_length, max_length, NORM_METHOD)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: main(trial, dataset_name=dataset_name, user_num=user_num, top_n_item = top_n_freq), n_trials=100)

        logging.info('------------- end top_n_item is: {} -----------------'.format(top_n_freq))
        logging.info('------------- end top_n_item is: {} -----------------'.format(top_n_freq))
        logging.info('------------- end top_n_item is: {} -----------------'.format(top_n_freq))

