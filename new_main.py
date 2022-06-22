#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 4:32
# @Author : ZM7
# @File : new_main
# @Software: PyCharm

import datetime
import torch
from sys import exit
import pandas as pd
import numpy as np
from DGSR import DGSR, collate, collate_test
import dgl
import pickle
from utils import StaticData
import warnings
import argparse
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger
from new_data import generate_graph, save_graphs, generate_data, refine_time
from utils import user_neg


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='Beauty', help='data name: sample')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
parser.add_argument('--item_max_length', type=int, default=50, help='the max length of item sequence')
parser.add_argument('--user_max_length', type=int, default=50, help='the max length of use sequence')
parser.add_argument('--k_hop', type=int, default=2, help='sub-graph size')
parser.add_argument('--gpu', default='3')
parser.add_argument("--record", action='store_true', default=False, help='record experimental results')
parser.add_argument("--val", action='store_true', default=False)
parser.add_argument("--model_record", action='store_true', default=False, help='record model')

opt = parser.parse_args()
args, extras = parser.parse_known_args()
device = torch.device(f'cuda:{opt.gpu}')
torch.cuda.set_device(device)
print(f"device: {device}")
print(f"opt: {opt}")

if opt.record:
    log_file = f'results/{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
               f'_layer_{opt.layer_num}_l2_{opt.l2}'
    mkdir_if_not_exist(log_file)
    sys.stdout = Logger(log_file)
    print(f'Logging to {log_file}')
if opt.model_record:
    model_file = f'{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
               f'_layer_{opt.layer_num}_l2_{opt.l2}'

# loading data (and preprocessing if necessary)
data_path = f'static/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/'
train_path = data_path + 'train/'
test_path = data_path + 'test/'
val_path = data_path + 'val/'
graph_path = data_path + 'graph'
metadata_path = data_path + 'meta'
def preprocess():
    print('start preprocessing:', datetime.datetime.now(), flush=True)
    for path in [data_path, train_path, test_path, val_path]:
        mkdir_if_not_exist(path)
    data = pd.read_csv('./data/' + opt.data + '.csv')
    data = data.groupby('user_id').apply(refine_time).reset_index(drop=True)
    
    # metadata
    metadata = {key + '_num': len(data[key + '_id'].unique()) for key in ['user', 'item']}
    
    # negative samples
    data_neg = user_neg(data, metadata['item_num'])
    with open(data_path + 'neg', 'wb') as file:
        pickle.dump(data_neg, file)
    
    # graph
    if not os.path.exists(graph_path):
        graph = generate_graph(data)
        save_graphs(graph_path, graph)
    else:
        graph = dgl.load_graphs(graph_path)[0][0]
        
    # data
    train_num, test_num = generate_data(data, graph, opt.item_max_length, opt.user_max_length, 
                                        train_path, test_path, val_path, job=opt.epoch, k_hop=opt.k_hop)
    
    # save metadata
    with open(metadata_path, 'wb') as file:
        pickle.dump(metadata, file)
        
    print('The number of train set:', train_num, flush=True)
    print('The number of test set:', test_num, flush=True)
    print('end preprocessing:', datetime.datetime.now(), flush=True)
if not os.path.exists(metadata_path):
    preprocess()
with open(metadata_path, 'rb') as file:
    metadata = pickle.load(file)
    user_num = metadata['user_num']
    item_num = metadata['item_num']
    


train_set = StaticData(train_path, dgl.load_graphs)
test_set = StaticData(test_path, dgl.load_graphs)
if opt.val:
    val_set = StaticData(val_path, dgl.load_graphs)

print('train number:', train_set.size)
print('test number:', test_set.size)
print('user number:', user_num)
print('item number:', item_num)
with open('results/'+opt.data+'_neg', 'rb') as f:
    data_neg = pickle.load(f) # 用于评估测试集
train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True, pin_memory=True, num_workers=12)
test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=8)
if opt.val:
    val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=2)

# 初始化模型
model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
             user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, 
             layer_num=opt.layer_num).cuda()
print(next(model.parameters()).device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
loss_func = nn.CrossEntropyLoss()
best_result = [0, 0, 0, 0, 0, 0]   # hit5,hit10,hit20,mrr5,mrr10,mrr20
best_epoch = [0, 0, 0, 0, 0, 0]
stop_num = 0
for epoch in range(opt.epoch):
    stop = True
    epoch_loss = 0
    iter = 0
    print('start training: ', datetime.datetime.now())
    model.train()
    for user, batch_graph, label, last_item in train_data:
        iter += 1
        score = model(batch_graph.to(device), user.cuda(), last_item.cuda(), is_training=True)
        loss = loss_func(score, label.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if iter % 400 == 0:
            print('Iter {}, loss {:.4f}'.format(iter, epoch_loss/iter), datetime.datetime.now())
    epoch_loss /= iter
    model.eval()
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), '=============================================')

    # val
    if opt.val:
        print('start validation: ', datetime.datetime.now())
        val_loss_all, top_val = [], []
        with torch.no_grad:
            for user, batch_graph, label, last_item, neg_tar in val_data:
                score, top = model(batch_graph.to(device), user.cuda(), last_item.cuda(), neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).cuda(), is_training=False)
                val_loss = loss_func(score, label.cuda())
                val_loss_all.append(val_loss.append(val_loss.item()))
                top_val.append(top.detach().cpu().numpy())
            recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(top_val)
            print('train_loss:%.4f\tval_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
                  '\tNDGG10@10:%.4f\tNDGG@20:%.4f' %
                  (epoch_loss, np.mean(val_loss_all), recall5, recall10, recall20, ndgg5, ndgg10, ndgg20))

    # test
    print('start predicting: ', datetime.datetime.now())
    all_top, all_label, all_length = [], [], []
    all_loss = []
    iter = 0
    for user, batch_graph, label, last_item, neg_tar in test_data:
        iter += 1
        with torch.no_grad():
            score, top = model(batch_graph.to(device), user.cuda(), last_item.cuda(), neg_tar=torch.cat([label.unsqueeze(1), neg_tar],-1).cuda(),  is_training=False)
            test_loss = loss_func(score, label.cuda())
            all_loss.append(test_loss.item())
            all_top.append(top.detach().cpu().numpy())
            all_label.append(label.numpy())
            if (iter + 1) % 200 == 0:
                print('Iter {}, test_loss {:.4f}'.format(iter + 1, np.mean(all_loss)), datetime.datetime.now())
        recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(all_top)
        if recall5 > best_result[0]:
            best_result[0] = recall5
            best_epoch[0] = epoch
            stop = False
        if recall10 > best_result[1]:
            if opt.model_record:
                torch.save(model.state_dict(), 'save_models/'+ model_file + '.pkl')
            best_result[1] = recall10
            best_epoch[1] = epoch
            stop = False
        if recall20 > best_result[2]:
            best_result[2] = recall20
            best_epoch[2] = epoch
            stop = False
            # ------select Mrr------------------
        if ndgg5 > best_result[3]:
            best_result[3] = ndgg5
            best_epoch[3] = epoch
            stop = False
        if ndgg10 > best_result[4]:
            best_result[4] = ndgg10
            best_epoch[4] = epoch
            stop = False
        if ndgg20 > best_result[5]:
            best_result[5] = ndgg20
            best_epoch[5] = epoch
            stop = False
        if stop:
            stop_num += 1
        else:
            stop_num = 0
        print('train_loss:%.4f\ttest_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
              '\tNDGG10@10:%.4f\tNDGG@20:%.4f\tEpoch:%d,%d,%d,%d,%d,%d' %
              (epoch_loss, np.mean(all_loss), best_result[0], best_result[1], best_result[2], best_result[3],
               best_result[4], best_result[5], best_epoch[0], best_epoch[1],
               best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5]))
    print(iter)