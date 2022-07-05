#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 4:32
# @Author : ZM7
# @File : new_main
# @Software: PyCharm

import datetime
import torch
import pandas as pd
import numpy as np
from DGSR import DGSR
import dgl
import pickle
from utils import StaticData
import warnings
import argparse
import os
import sys
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from utils import user_neg, eval_metric, mkdir_if_not_exist, collate, collate_test, get_multihot_label
from preprocess import generate_graph, save_graphs, generate_data, preprocess_data


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='Beauty', help='data name: sample')
parser.add_argument('--load', type=str, default=None, help='past model to load (default: from scratch)')
parser.add_argument('--batchSize', type=int, default=1024, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
parser.add_argument("--use_hinge", action='store_true', default=False, help='use multiclass hinge instead of cross entropy')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
parser.add_argument('--item_max_length', type=int, default=50, help='the max length of item sequence')
parser.add_argument('--user_max_length', type=int, default=50, help='the max length of use sequence')
parser.add_argument('--k_hop', type=int, default=2, help='sub-graph size')
parser.add_argument('--gpu', default='2')
parser.add_argument("--val", action='store_true', default=False)
parser.add_argument("--debug", action='store_true', default=False, help='debug mode (model not saved)')


opt = parser.parse_args()
args, extras = parser.parse_known_args()
device = torch.device(f'cuda:{opt.gpu}')
torch.cuda.set_device(device)
print(f"device: {device}")
print(f"opt: {opt}")

# loading data (and preprocessing if necessary)
data_id = f"{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}_{opt.use_hinge}"
run_id = f"bs{opt.batchSize}_lr{opt.lr}_epoch{opt.epoch}"
data_path = 'static/' + data_id + '/'
train_path = data_path + 'train/'
test_path = data_path + 'test/'
val_path = data_path + 'val/'
graph_path = data_path + 'graph'
metadata_path = data_path + 'meta'
neg_path = data_path + 'neg'
reverse_path = data_path + 'reverse'
out_folder = 'results/' + data_id + '/'
mkdir_if_not_exist(out_folder)
out_file = out_folder + run_id + '.out'
model_file = out_folder + 'model_' + run_id
sys.stdout = open(out_file, 'w+')
def preprocess():
    print('start preprocessing:', datetime.datetime.now(), flush=True)
    for path in [data_path, train_path, test_path, val_path]:
        mkdir_if_not_exist(path)
    data = pd.read_csv('./data/' + opt.data + '.csv')
    
    # refine user, item, and time indices
    data, u_rev, i_rev = preprocess_data(data)
    
    # mappings back to original user/item ids
    with open(reverse_path, 'wb') as file:
        pickle.dump({'u_rev': u_rev, 'i_rev': i_rev}, file)
    
    # metadata
    metadata = {key + '_num': len(data[key + '_id'].unique()) for key in ['user', 'item']}
    
    # negative samples
    data_neg = user_neg(data, metadata['item_num'])
    with open(neg_path, 'wb') as file:
        pickle.dump(data_neg, file)
    
    # graph
    if not os.path.exists(graph_path):
        graph = generate_graph(data)
        save_graphs(graph_path, graph)
    else:
        graph = dgl.load_graphs(graph_path)[0][0]
        
    # data
    train_num, test_num = generate_data(data, graph, metadata['item_num'], opt.item_max_length, opt.user_max_length, 
                                        train_path, test_path, val_path, job=opt.epoch, k_hop=opt.k_hop, use_hinge=opt.use_hinge)
    
    # save metadata (last to indicate completion)
    with open(metadata_path, 'wb') as file:
        pickle.dump(metadata, file)
        
    print('The number of train set:', train_num, flush=True)
    print('The number of test set:', test_num, flush=True)
    print('end preprocessing:', datetime.datetime.now(), flush=True)
if not os.path.exists(metadata_path):
    preprocess()
else:
    print("skipped preprocessing", flush=True)
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
with open(neg_path, 'rb') as f:
    data_neg = pickle.load(f) # negatives for evaluation
train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True, pin_memory=True, num_workers=12)
test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=8)
if opt.val:
    val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=2)

# initialize the model
model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
             user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, 
             layer_num=opt.layer_num).cuda()
if opt.load:
    state = torch.load(out_folder + 'model_' + opt.load)
    model.load_state_dict(state)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
loss_func = nn.MultiLabelMarginLoss() if opt.use_hinge else nn.BCEWithLogitsLoss()
best = {}
stop_num = 0
epoch_start = None
for epoch in range(opt.epoch):
    stop = True
    epoch_loss = 0
    iter = 0
    
    ############################ train ############################
    curr_time = datetime.datetime.now()
    if epoch_start is not None:
        print('last epoch time: ', curr_time - epoch_start)
        print('projected end time: ', curr_time + (curr_time - epoch_start) * (opt.epoch - epoch))
    print('start training: ', curr_time)
    epoch_start = curr_time
    model.train()
    for user, batch_graph, label, last_item in train_data:
        iter += 1
        score = model(batch_graph.to(device), user.cuda(), last_item.cuda(), is_training=True)
        print(label.dtype, flush=True)
        loss = loss_func(score, label.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if iter % 400 == 0:
            print('Iter {}, loss {:.4f}'.format(iter, epoch_loss/iter), datetime.datetime.now())
    epoch_loss /= iter
    ###############################################################

    ############################ val ##############################
    model.eval()
    if opt.val:
        print('start validation: ', datetime.datetime.now())
        val_losses, all_scores, neg_idxs = [], [], []
        with torch.no_grad:
            for user, batch_graph, label, last_item, all, neg_idx in val_data:
                score, all_score = model(batch_graph.to(device), user.cuda(), last_item.cuda(), neg_tar=all.cuda(), is_training=False)
                val_loss = loss_func(score, label.cuda())
                val_losses.append(val_loss.item())
                all_scores.append(all_score.detach().cpu().numpy())
                neg_idxs.append(neg_idx.numpy())
            results = eval_metric(all_scores, neg_idxs)
            results_str = sum([f"\n\t{metric_name}: {val:.4f}" for metric_name, val in results])
            print(f"\ttrain_loss:{epoch_loss:.4f} \
                   \n\tval_loss:{np.mean(val_losses):.4f}" + results_str)
    ###############################################################
    
    ############################ test ############################
    print('start testing: ', datetime.datetime.now())
    losses, all_scores, neg_idxs = [], [], []
    iter = 0
    for user, batch_graph, label, last_item, all, neg_idx in test_data:
        iter += 1
        with torch.no_grad():
            score, all_score = model(batch_graph.to(device), user.cuda(), last_item.cuda(), neg_tar=all.cuda(), is_training=False)
            test_loss = loss_func(score, label.cuda())
            losses.append(test_loss.item())
            all_scores.append(all_score.detach().cpu().numpy())
            neg_idxs.append(neg_idx.numpy())
            if (iter + 1) % 50 == 0:
                print('\tIter {}, test_loss {:.4f}'.format(iter + 1, np.mean(losses)), datetime.datetime.now())
        results = eval_metric(all_scores, neg_idxs)
        for metric_name, val in results.items():
            if metric_name not in best or val > best[metric_name][0]:
                stop = metric_name not in best   # stop=False when val exceeds best_result
                best[metric_name] = (val, epoch)
        if stop:
            stop_num += 1
        else:
            stop_num = 0
    results_str = sum([f"\n\t{metric_name} at {epoch}: {val:.4f}" for metric_name, (val, epoch) in best])
    print(f"\ttrain_loss:{epoch_loss:.4f} \
            \n\ttest_loss:{np.mean(losses):.4f}" + results_str)
    print('Epoch {}'.format(epoch), '=============================================')
    ###############################################################
sys.stdout.close()