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
from DGNN import DGNN
import dgl
import pickle
from utils import SamplingGraphData, GraphData
import warnings
import argparse
import os
import sys
from torch.utils.data import DataLoader
from torch.distributed.pipeline.sync import Pipe
from torch.distributed import rpc
import torch.optim as optim
import torch.nn as nn
from utils import user_neg, eval_metric, mkdir_if_not_exist, get_collate, get_topk_items
from preprocess import generate_graph, save_graphs, generate_data, preprocess_data


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='Enrollments', help='data name: sample')
parser.add_argument('--load', type=str, default=None, help='past model to load (default: from scratch)')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--multiplier', type=int, default=1, help='Number of graph samples per bucket (only applies if --sampling)')
parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
parser.add_argument("--test_num", type=int, default=4, help='Number of test times')
parser.add_argument("--k_hop", type=int, default=3, help='Number of hops in preprocessing')
parser.add_argument("--margin", action='store_true', default=False, help='Use multilabel margin loss')
parser.add_argument("--sampling", action='store_true', default=False, help='Whether to subsample input graphs')
parser.add_argument("--use_item_feat", action='store_true', default=False, help='Whether to use the item input embedding to calculate scores')
parser.add_argument("--pos_weight", type=float, default=1.0, help='Weighting on positive examples')
parser.add_argument("--max_users", type=int, default=25, help='Maximum number of sampled users per hop')
parser.add_argument("--max_items", type=int, default=25, help='Maximum number of sampled items per hop')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
parser.add_argument('--feat_drop', type=float, default=0.0, help='drop_out')
parser.add_argument('--attn_drop', type=float, default=0.2, help='drop_out')
parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
parser.add_argument('--neg_num', type=int, default=100, help='Number of negatives to sample')
parser.add_argument('--max_lookback', type=int, default=12, help='maximum lookback in original time')
parser.add_argument('--gpu', default='2')
parser.add_argument("--val", action='store_true', default=False)
parser.add_argument("--nosave", action='store_true', default=False, help='model and outputs not saved')
parser.add_argument("--run_id", type=str, default='', help='Additional identifier for run (outside of hparams)')
parser.add_argument("--port", type=str, default='29500', help='RPC Port')

opt = parser.parse_args()
args, extras = parser.parse_known_args()
devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]     # get available devices
if len(devices) > opt.layer_num + 2:                                                # use at most layer_num + 2 devices
    devices = devices[:opt.layer_num + 2]
device = devices[-1]                                                                # device to embed and predict on
devices = devices[:-1]                                                              # devices to compute graph layers on
torch.cuda.set_device(device)                                                       # redirect .cuda() to correct device
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = opt.port
rpc.init_rpc('worker', rank=0, world_size=1)
print(f"devices: {devices}")
print(f"opt: {opt}")

# loading data (and preprocessing if necessary)
data_id = f"{opt.data}_{opt.test_num}_{opt.max_lookback}_{opt.val}"
run_id = f"bs{opt.batch_size}_lr{opt.lr}_ep{opt.epoch}_l2{opt.l2}_pw{opt.pos_weight}_ft{opt.feat_drop}_at{opt.attn_drop}_ln{opt.layer_num}_hs{opt.hidden_size}"
if opt.sampling:
    run_id += f"_mu{opt.max_users}_mi{opt.max_items}_k{opt.k_hop}"
run_id += '_' + ('margin' if opt.margin else 'bce')
run_id += '_' + ('feat' if opt.use_item_feat else 'embed')
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
results_path = data_path + run_id + '/'
mkdir_if_not_exist(results_path)
model_file = results_path + 'model'
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
    metadata = {**metadata, 'etypes': graph.canonical_etypes}
        
    # data
    print('start data generation:', datetime.datetime.now(), flush=True)
    path_args = (train_path, test_path, val_path)
    train_num, val_num, test_num = generate_data(data, graph, opt.max_lookback, *path_args, opt.test_num)
    
    # save metadata (last to indicate completion)
    with open(metadata_path, 'wb') as file:
        pickle.dump(metadata, file)
        
    print('The number of train set: ', train_num // opt.batch_size, flush=True)
    print('The number of val set: ', val_num // opt.batch_size, flush=True)
    print('The number of test set: ', test_num // opt.batch_size, flush=True)
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

Dataset = SamplingGraphData if opt.sampling else GraphData
train_set = Dataset(train_path)
test_set = Dataset(test_path)
if opt.val:
    val_set = Dataset(val_path)

batch_size = opt.batch_size // (opt.multiplier if opt.sampling else 1)
find_num_batches = lambda size: size // batch_size + (size % batch_size > 0)
find_log_freq = lambda n, k: 1 if n < k else 2 * find_log_freq(n / 2, k)     # log freq so that we log at least k/2 and at most k times
train_num = find_num_batches(train_set.size)
test_num = find_num_batches(test_set.size)
train_log_freq = find_log_freq(train_num, 20)
test_log_freq = find_log_freq(test_num, 10)
print('train number: ', train_num)
print('test number: ', test_num)
print('user number: ', user_num)
print('item number: ', item_num)
with open(neg_path, 'rb') as f:
    data_neg = pickle.load(f) # negatives for evaluation
collate = get_collate(item_num, opt.max_users, opt.max_items, opt.k_hop, True,
                      opt.margin, opt.sampling, opt.multiplier)
collate_test = get_collate(item_num, opt.max_users, opt.max_items, opt.k_hop, False,
                           opt.margin, opt.sampling, opt.multiplier)
train_data = DataLoader(dataset=train_set, 
                        batch_size=batch_size, 
                        collate_fn=collate, 
                        shuffle=True, 
                        pin_memory=True, 
                        num_workers=30)
test_data = DataLoader(dataset=test_set, 
                       batch_size=batch_size, 
                       collate_fn=collate_test, 
                       shuffle=True, 
                       pin_memory=True, 
                       num_workers=12)
if opt.val:
    val_data = DataLoader(dataset=val_set, 
                          batch_size=opt.batch_size // opt.multiplier, 
                          collate_fn=collate_test, 
                          pin_memory=True, 
                          num_workers=2)

# initialize the model
args = (etypes, {'user': user_num, 'item': item_num}, opt.hidden_size, opt.max_lookback, 
        device, devices, opt.feat_drop, opt.attn_drop, opt.layer_num, opt.use_item_feat)
model = Pipe(DGNN(*args), min(opt.batch_size, 8), 'never')
# model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, max_lookback=opt.max_lookback, 
#              feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, layer_num=opt.layer_num).cuda()
if opt.load:
    state = torch.load(data_path + 'model_' + opt.load)
    model.load_state_dict(state)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
Loss = nn.MultiLabelMarginLoss if opt.margin else nn.BCEWithLogitsLoss
kwargs = {} if opt.margin else {'pos_weight': torch.tensor(opt.pos_weight)}
loss_func = Loss(reduction='mean', **kwargs).to(device)
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
    for batch_graph, user, batch_idx, target in train_data:
        iter += 1
        batch_graph = batch_graph.to(device)
        user = user.cuda()
        batch_idx = batch_idx.cuda()
        target = target.cuda()
        if opt.use_item_feat:
            score, item_idx = model(batch_graph, user, batch_idx).local_value()
        else:
            score = model(batch_graph, user, batch_idx).local_value()
            item_idx = torch.arange(item_num, device=device)
        loss = loss_func(score, target[:, item_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().cpu().item()
        if iter % train_log_freq == 0:
            if not opt.nosave:
                torch.save(score, results_path + f"score_{epoch}_{iter}")
            print('\tIter {}, loss {:.4f}'.format(iter, epoch_loss/iter), datetime.datetime.now(), flush=True)
        del batch_graph, user, target, loss, score
        torch.cuda.empty_cache()
    epoch_loss /= iter
    ###############################################################

    ############################ val ##############################
    model.eval()
    iter = 0
    if opt.val:
        print('start validation: ', datetime.datetime.now())
        top_items, sample_top_items, num_targets, labels = [], [], [], []
        ats = [5, 10, 20]
        with torch.no_grad:
            for batch_graph, user, batch_idx, target, label, num_target in val_data:
                iter += 1
                batch_graph = batch_graph.to(device)
                user = user.cuda()
                batch_idx = batch_idx.cuda()
                target = target.cuda()
                label = label.cuda()
                num_target = num_target.cuda()
                if opt.use_item_feat:
                    score, item_idx = model(batch_graph, user, batch_idx).local_value()
                else:
                    score = model(batch_graph, user, batch_idx).local_value()
                    item_idx = torch.arange(item_num, device=device)
                target = target[:, item_idx]
                loss = loss_func(score, target)
                val_loss += loss.detach().cpu().item()
                # score = score.reshape(-1, opt.multiplier, item_num).mean(1)
                top, sample_top = get_topk_items(score, item_idx, target, num_target, max(ats), opt.neg_num)
                top_items.append(top)
                sample_top_items.append(sample_top)
                num_targets.append(num_target)
                labels.append(label)
                if iter % (test_set.size // 1000) == 0:
                    print('\tIter {}'.format(iter), datetime.datetime.now(), flush=True)
                if not opt.nosave:
                    torch.save(score, results_path + f"val_score_{epoch}_{iter}")
                del batch_graph, user, score, target, loss
                torch.cuda.empty_cache()
            label = torch.cat(labels, 0)
            top_item = torch.cat(top_items, 0)
            sample_top_item = torch.cat(sample_top_items, 0)
            num_target = torch.cat(num_targets, 0)
            for name, top in zip(["true", "sampling"], [top_item, sample_top_item]):
                results = eval_metric(top, label, num_target, ats)
                results_str = '\n\t'.join([f"{metric_name}: {val:.4f}" for metric_name, val in results.items()])
                print(f"{name} results:\n\t" + results_str, flush=True)
    ###############################################################
    
    ############################ test ############################
    print('start testing: ', datetime.datetime.now())
    top_items, sample_top_items, num_targets, labels = [], [], [], []
    iter = 0
    ats = [5, 10, 20]
    with torch.no_grad():
        for batch_graph, user, batch_idx, target, label, num_target in test_data:
            iter += 1
            batch_graph = batch_graph.to(device)
            user = user.cuda()
            batch_idx = batch_idx.cuda()
            target = target.cuda()
            label = label.cuda()
            num_target = num_target.cuda()
            if opt.use_item_feat:
                score, item_idx = model(batch_graph, user, batch_idx).local_value()
            else:
                score = model(batch_graph, user, batch_idx).local_value()
                item_idx = torch.arange(item_num, device=device)
            target = target[:, item_idx]
            loss = loss_func(score, target)
            test_loss += loss.detach().cpu().item()
            # score = score.reshape(-1, opt.multiplier, item_num).mean(1)
            top, sample_top = get_topk_items(score, item_idx, target, num_target, max(ats), opt.neg_num)
            top_items.append(top)
            sample_top_items.append(sample_top)
            num_targets.append(num_target)
            labels.append(label)
            if iter % test_log_freq == 0:
                if not opt.nosave:
                    torch.save(score, results_path + f"test_score_{epoch}_{iter}")
                print('\tIter {}, test_loss {:.4f}'.format(iter, test_loss / iter), datetime.datetime.now(), flush=True)
            del batch_graph, user, target, score, loss
            torch.cuda.empty_cache()
        label = torch.cat(labels, 0)
        top_item = torch.cat(top_items, 0)
        sample_top_item = torch.cat(sample_top_items, 0)
        num_target = torch.cat(num_targets, 0)
        for name, top in zip(["true", "sampling"], [top_item, sample_top_item]):
            results = eval_metric(top, label, num_target, ats)
            results_str = '\n\t'.join([f"{metric_name}: {val:.4f}" for metric_name, val in results.items()])
            print(f"{name} results:\n\t" + results_str, flush=True)
        print('Epoch {}'.format(epoch + 1), '=============================================')
    for metric_name, val in results.items():
        if metric_name not in best or val > best[metric_name][0]:
            stop = metric_name not in best   # stop=False when val exceeds best_result
            best[metric_name] = (val, epoch)
        if epoch == best[metric_name][1] and not opt.nosave:
            file_add = '' if metric_name == 'recall@10' else f'_{metric_name}'
            torch.save(model.state_dict(), model_file + file_add)
    if stop:
        stop_num += 1
    else:
        stop_num = 0
    
    ###############################################################
    print(f"max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.4f}")
    print(f"max memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.4f}")
    print(f"max memory cached: {torch.cuda.max_memory_cached() / 1e9:.4f}")
results_str = '\n\t'.join([f"{metric_name} at {epoch}: {val:.4f}" for metric_name, (val, epoch) in best.items()])
print(f"\t{results_str}")
sys.stdout.close()