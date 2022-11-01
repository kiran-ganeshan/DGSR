#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 4:32
# @Author : ZM7
# @File : new_main
# @Software: PyCharm

import argparse
import datetime
import os
import pickle
import sys
import warnings

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed import rpc
from torch.distributed.pipeline.sync import Pipe
from torch.utils.data import DataLoader

from DGNN import DGNN
from preprocess import preprocess
from utils import (GraphData, SamplingGraphData, eval_metric, get_collate,
                   get_topk_items, mkdir_if_not_exist)

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='Enrollments', help='data name: sample')
parser.add_argument('--load', type=str, default=None, help='past model to load (default: from scratch)')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
parser.add_argument("--test_num", type=int, default=4, help='Number of test times')
parser.add_argument("--k_hop", type=int, default=3, help='Number of hops in preprocessing')
parser.add_argument("--sampling", action='store_true', default=False, help='Whether to subsample input graphs')
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
parser.add_argument("--val", action='store_true', default=False)
parser.add_argument("--nosave", action='store_true', default=False, help='model and outputs not saved')
parser.add_argument("--run_id", type=str, default='', help='Additional identifier for run (outside of hparams)')
parser.add_argument("--device", type=int, default=3, help='Device to Use')

opt = parser.parse_args()
device = torch.device(f'cuda:{opt.device}')
torch.cuda.set_device(device) 
print(f"devices: {devices}")
print(f"opt: {opt}")

# loading data (and preprocessing if necessary)
data_id = f"{opt.data}_{opt.test_num}_{opt.max_lookback}_{opt.val}"
run_id = f"bs{opt.batch_size}_lr{opt.lr}_ep{opt.epoch}_l2{opt.l2}_pw{opt.pos_weight}_ft{opt.feat_drop}_at{opt.attn_drop}_ln{opt.layer_num}_hs{opt.hidden_size}"
if opt.sampling:
    run_id += f"_mu{opt.max_users}_mi{opt.max_items}_k{opt.k_hop}"
if opt.run_id:
    run_id = opt.run_id + '_' + run_id
data_path = 'static/' + data_id + '/'
out_folder = 'results/' + data_id + '/'
mkdir_if_not_exist(out_folder)
out_file = out_folder + run_id + '.out'
results_path = data_path + run_id + '/'
mkdir_if_not_exist(results_path)
model_file = results_path + 'model'
sys.stdout = open(out_file, 'w+')
train_path, test_path, val_path, num_nodes, etypes, ntypes = preprocess(opt, data_path)
Dataset = SamplingGraphData if opt.sampling else GraphData
train_set = Dataset(train_path)
test_set = Dataset(test_path)
if opt.val:
    val_set = Dataset(val_path)

find_num_batches = lambda size: size // opt.batch_size + (size % opt.batch_size > 0)
find_log_freq = lambda n, k: 1 if n < k else 2 * find_log_freq(n / 2, k)     # log freq so that we log at least k/2 and at most k times
train_num = find_num_batches(train_set.size)
test_num = find_num_batches(test_set.size)
train_log_freq = find_log_freq(train_num, 20)
test_log_freq = find_log_freq(test_num, 10)
print('number of training batches: ', train_num)
print('number of testing batches: ', test_num)
for ntype in ntypes:
    print(f'{ntype} number: ', num_nodes[ntype])
collate = get_collate(num_nodes['item'], opt.max_users, opt.max_items, 
                      opt.k_hop, True, opt.sampling)
collate_test = get_collate(num_nodes['item'], opt.max_users, opt.max_items, 
                           opt.k_hop, False, opt.sampling)
train_data = DataLoader(dataset=train_set, 
                        batch_size=opt.batch_size, 
                        collate_fn=collate, 
                        shuffle=True, 
                        pin_memory=True, 
                        num_workers=30)
test_data = DataLoader(dataset=test_set, 
                       batch_size=opt.batch_size, 
                       collate_fn=collate_test, 
                       shuffle=True, 
                       pin_memory=True, 
                       num_workers=12)
if opt.val:
    val_data = DataLoader(dataset=val_set, 
                          batch_size=opt.batch_size, 
                          collate_fn=collate_test, 
                          pin_memory=True, 
                          num_workers=2)

# initialize the model
args = (etypes, ntypes, num_nodes, opt.hidden_size, opt.max_lookback, device, devices, 
        opt.feat_drop, opt.attn_drop, opt.layer_num)
model = Pipe(DGNN(*args), min(opt.batch_size, 8), 'never')
# model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, max_lookback=opt.max_lookback, 
#              feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, layer_num=opt.layer_num).cuda()
if opt.load:
    state = torch.load(data_path + 'model_' + opt.load)
    model.load_state_dict(state)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
loss_func = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(opt.pos_weight)).to(device)
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
        # print(user.device, batch_idx.device, target.device)
        score = model(batch_graph, user).local_value()
        loss = loss_func(score, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().cpu().item()
        if iter % train_log_freq == 0:
            print('\tIter {}, loss {:.4f}'.format(iter, epoch_loss/iter), datetime.datetime.now(), flush=True)
        del batch_graph, user, target, loss, score
        torch.cuda.empty_cache()
    epoch_loss /= iter
    ###############################################################
    print([torch.norm(embed) for name, embed in model[1].named_parameters() if name == 'module.embeds.item.weight'], flush=True)
    ############################ val ##############################
    model.eval()
    iter = 0
    if opt.val:
        print('start validation: ', datetime.datetime.now())
        top_items, sample_top_items, num_targets, labels = [], [], [], []
        ats = [5, 10, 20]
        with torch.no_grad:
            for batch_graph, user, target, label, num_target in val_data:
                iter += 1
                batch_graph = batch_graph.to(device)
                user = user.cuda()
                target = target.cuda()
                label = label.cuda()
                num_target = num_target.cuda()
                score = model(batch_graph, user).local_value()
                loss = loss_func(score, target)
                val_loss += loss.detach().cpu().item()
                top, sample_top = get_topk_items(score, target, num_target, max(ats), opt.neg_num)
                top_items.append(top)
                sample_top_items.append(sample_top)
                num_targets.append(num_target)
                labels.append(label)
                if iter % (test_set.size // 1000) == 0:
                    print('\tIter {}'.format(iter), datetime.datetime.now(), flush=True)
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
        for batch_graph, user, target, label, num_target in test_data:
            iter += 1
            batch_graph = batch_graph.to(device)
            user = user.cuda()
            target = target.cuda()
            label = label.cuda()
            num_target = num_target.cuda()
            score = model(batch_graph, user).local_value()
            loss = loss_func(score, target)
            test_loss += loss.detach().cpu().item()
            top, sample_top = get_topk_items(score, target, num_target, max(ats), opt.neg_num)
            top_items.append(top)
            sample_top_items.append(sample_top)
            num_targets.append(num_target)
            labels.append(label)
            if iter % test_log_freq == 0:
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