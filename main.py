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
from utils import user_neg, eval_metric, mkdir_if_not_exist, get_collate
from preprocess import generate_graph, save_graphs, generate_data, preprocess_data


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='Enrollments', help='data name: sample')
parser.add_argument('--load', type=str, default=None, help='past model to load (default: from scratch)')
parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
parser.add_argument("--test_num", type=int, default=4, help='Number of test times')
parser.add_argument("--k_hop", type=int, default=3, help='Number of hops in preprocessing')
parser.add_argument("--max_users", type=int, default=25, help='Maximum number of sampled users per hop')
parser.add_argument("--max_items", type=int, default=5, help='Maximum number of sampled items per hop')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
parser.add_argument('--feat_drop', type=float, default=0.0, help='drop_out')
parser.add_argument('--attn_drop', type=float, default=0.0, help='drop_out')
parser.add_argument('--mean', action='store_true', default=False, help='Train to mean')
parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
parser.add_argument('--max_lookback', type=int, default=25, help='maximum lookback in original time')
parser.add_argument('--gpu', default='0')
parser.add_argument("--val", action='store_true', default=False)
parser.add_argument("--debug", action='store_true', default=False, help='debug mode (model not saved)')
parser.add_argument("--run_id", type=str, default='', help='Additional identifier for run (outside of hparams)')

opt = parser.parse_args()
args, extras = parser.parse_known_args()
device = torch.device(f'cuda:{opt.gpu}')
torch.cuda.set_device(device)
print(f"device: {device}")
print(f"opt: {opt}")

# loading data (and preprocessing if necessary)
data_id = f"{opt.data}_{opt.test_num}_{opt.max_lookback}"
run_id = f"bs{opt.batch_size}_lr{opt.lr}_ep{opt.epoch}_ft{opt.feat_drop}_at{opt.attn_drop}_ln{opt.layer_num}_hs{opt.hidden_size}_tm{opt.mean}_mu{opt.max_users}_mi{opt.max_items}_k{opt.k_hop}"
if opt.run_id:
    run_id = opt.run_id + '_' + run_id
data_path = 'static/' + data_id + '/'
train_path = data_path + 'train/'
test_path = data_path + 'test/'
val_path = data_path + 'val/' if opt.val else None
graph_path = data_path + 'graph'
metadata_path = data_path + 'meta'
neg_path = data_path + 'neg'
out_folder = 'results/' + data_id + '/'
mkdir_if_not_exist(out_folder)
out_file = out_folder + run_id + '.out'
model_file = data_path + 'model_' + run_id
sys.stdout = open(out_file, 'w+')
def preprocess():
    print('start preprocessing:', datetime.datetime.now(), flush=True)
    for path in [data_path, train_path, test_path, val_path]:
        if path:
            mkdir_if_not_exist(path)
    data = pd.read_csv('./data/' + opt.data + '.csv')
    
    # refine user, item, and time indices
    data, u_rev, i_rev = preprocess_data(data)
    
    # metadata
    metadata = {key + '_num': len(data[key + '_id'].unique()) for key in ['user', 'item']}
    metadata = {**metadata, 'user_rev': u_rev, 'item_rev': i_rev}
    
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
    metadata = {**metadata, 'ntypes': graph.ntypes, 'etypes': graph.canonical_etypes}
        
    # data
    print('start data generation:', datetime.datetime.now(), flush=True)
    train_num, val_num, test_num = generate_data(data, graph, metadata['item_num'], opt.max_lookback, 
                                                 train_path, test_path, val_path, opt.test_num, opt.k_hop)
    
    # save metadata (last to indicate completion)
    with open(metadata_path, 'wb') as file:
        pickle.dump(metadata, file)
        
    print('The number of train set: ', train_num, flush=True)
    print('The number of val set: ', val_num, flush=True)
    print('The number of test set: ', test_num, flush=True)
    print('End preprocessing: ', datetime.datetime.now(), flush=True)
if not os.path.exists(metadata_path):
    preprocess()
else:
    print("skipped preprocessing", flush=True)
with open(metadata_path, 'rb') as file:
    metadata = pickle.load(file)
    user_num = metadata['user_num']
    item_num = metadata['item_num']
    etypes = metadata['etypes']
    ntypes = metadata['ntypes']

train_set = StaticData(train_path)
test_set = StaticData(test_path)
if opt.val:
    val_set = StaticData(val_path)

print('train number: ', train_set.size)
print('test number: ', test_set.size)
print('user number: ', user_num)
print('item number: ', item_num)
with open(neg_path, 'rb') as f:
    data_neg = pickle.load(f) # negatives for evaluation
collate = get_collate(item_num, opt.max_users, opt.max_items, opt.k_hop, opt.mean)
collate_test = get_collate(item_num, opt.max_users, opt.max_items, opt.k_hop, opt.mean, data_neg)
train_data = DataLoader(dataset=train_set, batch_size=opt.batch_size, collate_fn=collate, shuffle=True, pin_memory=True, num_workers=30)
test_data = DataLoader(dataset=test_set, batch_size=opt.batch_size, collate_fn=collate_test, pin_memory=True, num_workers=12)
if opt.val:
    val_data = DataLoader(dataset=val_set, batch_size=opt.batch_size, collate_fn=collate_test, pin_memory=True, num_workers=2)

# initialize the model
# model = DGSR(etypes=etypes, ntypes=ntypes, user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, max_lookback=opt.max_lookback, 
#              feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, layer_num=opt.layer_num).cuda()
model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, max_lookback=opt.max_lookback, 
             feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, layer_num=opt.layer_num).cuda()
if opt.load:
    state = torch.load(data_path + 'model_' + opt.load)
    model.load_state_dict(state)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
loss_func = nn.BCEWithLogitsLoss(reduction='mean').to(device)
best = {}
stop_num = 0
epoch_start = None

for epoch in range(opt.epoch):
    stop = True
    epoch_loss = 0
    val_loss = 0
    test_loss = 0
    iter = 0
    
    ############################ train ############################
    curr_time = datetime.datetime.now()
    if epoch_start is not None:
        print('last epoch time: ', curr_time - epoch_start)
        print('projected end time: ', curr_time + (curr_time - epoch_start) * (opt.epoch - epoch))
    print('start training: ', curr_time)
    epoch_start = curr_time
    model.train()
    for batch_graph, user, target in train_data:
        iter += 1
        batch_graph = batch_graph.to(device)
        user = user.cuda()
        target = target.cuda()
        
        score = model(batch_graph, user, is_training=True)
        loss = loss_func(score, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().cpu().item()
        if iter % 1000 == 0:
            print('\tIter {}, loss {:.4f}'.format(iter, epoch_loss/iter), datetime.datetime.now(), flush=True)
    epoch_loss /= iter
    ###############################################################

    ############################ val ##############################
    model.eval()
    iter = 0
    if opt.val:
        print('start validation: ', datetime.datetime.now())
        ranks, neg_idxs = [], []
        with torch.no_grad:
            for batch_graph, user, target, all_label, num_pos in val_data:
                iter += 1
                batch_graph = batch_graph.to(device)
                user = user.cuda()
                target = target.cuda()
                all_label = all_label.cuda()
                num_pos = num_pos.cuda()
                score, all_score = model(batch_graph, user, all_label=all_label, is_training=False)
                loss = loss_func(score, target)
                val_loss += loss.cpu().item()
                rank = all_score.argsort(-1).argsort(-1)
                ranks.append(rank)
                neg_idxs.append(num_pos)
            ranks = [rank.detach().cpu().numpy() for rank in ranks]
            results = eval_metric(ranks, neg_idxs)
            results_str = '\n'.join([f"{metric_name}: {val:.4f}" for metric_name, val in results.items()])
            print(f"train_loss:{epoch_loss:.4f} \
                   \nval_loss:{(val_loss / iter):.4f} \
                   \n{results_str}", flush=True)
    ###############################################################
    
    ############################ test ############################
    print('start testing: ', datetime.datetime.now())
    ranks, neg_idxs = [], []
    iter = 0
    with torch.no_grad():
        for batch_graph, user, target, all_label, num_pos in test_data:
            iter += 1
            batch_graph = batch_graph.to(device)
            user = user.cuda()
            target = target.cuda()
            all_label = all_label.cuda()
            num_pos = num_pos.cuda()
            score, all_score = model(batch_graph, user, all_label=all_label, is_training=False)
            loss = loss_func(score, target)
            test_loss += loss.cpu().item()
            rank = all_score.argsort(-1).argsort(-1)
            ranks.append(rank)
            neg_idxs.append(num_pos)
            if iter % 200 == 0:
                print('\tIter {}, test_loss {:.4f}'.format(iter, test_loss / iter), datetime.datetime.now(), flush=True)
        ranks = torch.cat(ranks, 0)
        neg_idxs = torch.cat(neg_idxs, 0)
        results = eval_metric(ranks, neg_idxs, device)
    for metric_name, val in results.items():
        if metric_name not in best or val > best[metric_name][0]:
            stop = metric_name not in best   # stop=False when val exceeds best_result
            best[metric_name] = (val, epoch)
        if epoch == best[metric_name][1] and not opt.debug:
            file_add = '' if metric_name == 'recall@10' else f'_{metric_name}'
            torch.save(model.state_dict(), model_file + file_add)
    if stop:
        stop_num += 1
    else:
        stop_num = 0
    results_str = '\n'.join([f"{metric_name}: {val:.4f}" for metric_name, val in results.items()])
    print(f"train_loss:{epoch_loss:.4f} \
            \ntest_loss:{test_loss / iter:.4f} \
            \n{results_str}", flush=True)
    print('Epoch {}'.format(epoch + 1), '=============================================')
    ###############################################################
results_str = '\n\t'.join([f"{metric_name} at {epoch}: {val:.4f}" for metric_name, (val, epoch) in best.items()])
print(f"\t{results_str}")
sys.stdout.close()