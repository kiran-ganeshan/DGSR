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
parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
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
parser.add_argument("--nosave", action='store_true', default=False, help='model and outputs not saved')
parser.add_argument("--run_id", type=str, default='', help='Additional identifier for run (outside of hparams)')
parser.add_argument("--device", type=int, default=0, help='Device to Use')

opt = parser.parse_args()
device = torch.device(f'cuda:{opt.device}')
torch.cuda.set_device(device) 
print(f"devices: {device}")
print(f"opt: {opt}")

# loading data (and preprocessing if necessary)
data_id = f"{opt.data}_{opt.test_num}_{opt.max_lookback}_nopipe"
run_id = f"bs{opt.batch_size}_lr{opt.lr}_ep{opt.epoch}_l2{opt.l2}_"
run_id += f"pw{opt.pos_weight}_ft{opt.feat_drop}_at{opt.attn_drop}_"
run_id += f"ln{opt.layer_num}_hs{opt.hidden_size}"
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
splits, paths, num_nodes, etypes, ntypes = preprocess(opt, data_path)
assert len(set(splits).difference(paths.keys())) == 0       # splits list enumerates keys of paths dict
Dataset = SamplingGraphData if opt.sampling else GraphData
datasets = {split: Dataset(paths[split]) for split in splits}
collates = {split: get_collate(num_nodes['item'], opt.max_users, opt.max_items, 
                               opt.k_hop, split == 'train', opt.sampling) 
            for split in datasets.keys()}
loaders = {split: DataLoader(dataset=dataset, 
                             batch_size=opt.batch_size, 
                             shuffle=True, 
                             collate_fn=collates[split], 
                             pin_memory=True) 
           for split, dataset in datasets.items()}

find_num_batches = lambda size: size // opt.batch_size + (size % opt.batch_size > 0)
find_log_freq = lambda n, k: 1 if n < k else 2 * find_log_freq(n / 2, k)     # log freq so that we log at least k/2 and at most k times
num_batches = {split: find_num_batches(dataset.size) for split, dataset in datasets.items()}
log_freqs = {split: find_log_freq(dataset.size, 10) for split, dataset in datasets.items()}
for split, num_batch in num_batches.items():
    print(f'number of {split} batches: ', num_batch)
for ntype, num in num_nodes.items():
    print(f'{ntype} number: ', num)

# initialize the model
model = DGNN(etypes, ntypes, num_nodes, opt.hidden_size, opt.max_lookback, 
             device, opt.feat_drop, opt.attn_drop, opt.layer_num).cuda()
if opt.load:
    state = torch.load(data_path + 'model_' + opt.load)
    model.load_state_dict(state)
param_count = sum(p.numel() for p in model.parameters())
print('number of parameters: ', param_count)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
loss_kwargs = {'reduction': 'mean', 'pos_weight': torch.tensor(opt.pos_weight)}
loss_func = nn.BCEWithLogitsLoss(**loss_kwargs)#.to(device)

ats = [10]
def step(graph, user, target, label=None, num_target=None):
    graph = graph.to(device)
    user = user.cuda()
    score, item = model(graph, user)
    score, item = score.cpu(), item.cpu()
    target = target[:, item]
    loss = loss_func(score, target)
    if label is None or num_target is None:         # training
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.detach().item()
        top, sample_top = None, None
    else:                                           # evaluation
        top, sample_top = get_topk_items(score, item, target, num_target, max(ats), opt.neg_num)
        loss = loss.detach().cpu().item()
    del graph, user, target, score, item
    torch.cuda.empty_cache()
    return loss, top, sample_top

best = {}
epoch_start = None
for epoch in range(opt.epoch):
    
    # timing
    curr_time = datetime.datetime.now()
    if epoch_start is not None:
        print('last epoch time: ', curr_time - epoch_start)
        print('projected end time: ', curr_time + (curr_time - epoch_start) * (opt.epoch - epoch))
    epoch_start = curr_time
    
    for split in splits:
        iter, total_loss = 0, 0
        train = split == 'train'
        print(f'start {split}: ', datetime.datetime.now())
        if train:
            model.train()
        else:
            model.eval()
            tops, sample_tops, num_targets, labels = [], [], [], []
        for data in loaders[split]:
            iter += 1
            loss, top, sample_top = step(*data)
            if iter % log_freqs[split] == 0:
                print('\tIter {}, loss {:.4f}'.format(iter, loss / iter), datetime.datetime.now(), flush=True)
            total_loss += loss
            if not train:
                *others, label, num_target = data
                tops.append(top)
                sample_tops.append(sample_top)
                num_targets.append(num_target)
                labels.append(label)
        print('\ttotal loss {:.4f}'.format(total_loss / iter))
        if not train:
            label = torch.cat(labels, 0)
            top_item = torch.cat(tops, 0)
            sample_top_item = torch.cat(sample_tops, 0)
            num_target = torch.cat(num_targets, 0)
            for name, top in zip(["true", "sampling"], [top_item, sample_top_item]):
                results = eval_metric(top, label, num_target, ats)
                results_str = '\n\t\t'.join([f"{metric_name}: {val:.4f}" for metric_name, val in results.items()])
                print(f"\t{name} results:\n\t\t" + results_str, flush=True)
        if split == 'test':
            for metric_name, val in results.items():
                if metric_name not in best or val > best[metric_name][0]:
                    stop = metric_name not in best   # stop=False when val exceeds best_result
                    best[metric_name] = (val, epoch)
                if epoch == best[metric_name][1] and not opt.nosave:
                    file_add = '' if metric_name == 'recall@10' else f'_{metric_name}'
                    torch.save(model.state_dict(), model_file + file_add)
        
    print([torch.norm(embed) for name, embed in model.named_parameters() if name == 'embeds.item.weight'], flush=True)
    print(f"max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.4f}")
    print(f"max memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.4f}")
    print(f"max memory cached: {torch.cuda.max_memory_cached() / 1e9:.4f}")
results_str = '\n\t'.join([f"{metric_name} at {epoch}: {val:.4f}" for metric_name, (val, epoch) in best.items()])
print(f"\t{results_str}")
sys.stdout.close()