#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/15 7:30
# @Author : ZM7
# @File : new_data
# @Software: PyCharm

import dgl
import pandas as pd
import numpy as np
import datetime
import argparse
from dgl.sampling import sample_neighbors, select_topk
import torch
import os
from dgl import save_graphs
from joblib import Parallel, delayed

# Replace user and item ids with their relative orders
def order(data, key):
    user_ids = np.unique(data[key])
    order = {k: v for k, v in zip(user_ids, range(len(user_ids)))}
    reverse = {v: k for k, v in order.items()}
    data[key] = data[key].apply(lambda id: order[id])
    return data, reverse

def refine_time(data):
    data = data.sort_values(['time'], kind='mergesort')
    time_seq = data['time'].values
    time_gap = 1
    for i, da in enumerate(time_seq[0:-1]):
        if time_seq[i] == time_seq[i+1] or time_seq[i] > time_seq[i+1]:
            time_seq[i+1] = time_seq[i+1] + time_gap
            time_gap += 1
    data['time'] = time_seq
    return  data

def preprocess_data(data):
    data, u_reverse = order(data, 'user_id')
    data, i_reverse = order(data, 'item_id')
    data = data.groupby('user_id').apply(refine_time).reset_index(drop=True)
    return data, u_reverse, i_reverse

def generate_graph(data):
    user = data['user_id'].values
    item = data['item_id'].values
    time = data['time'].values
    graph_data = {('item','by','user'):(torch.tensor(item), torch.tensor(user)),
                  ('user','pby','item'):(torch.tensor(user), torch.tensor(item))}
    graph = dgl.heterograph(graph_data)
    graph.edges['by'].data['time'] = torch.LongTensor(time)
    graph.edges['pby'].data['time'] = torch.LongTensor(time)
    graph.nodes['user'].data['user_id'] = torch.LongTensor(np.unique(user))
    graph.nodes['item'].data['item_id'] = torch.LongTensor(np.unique(item))
    return graph


def generate_user(user, data, graph, item_max_length, user_max_length, train_path, test_path, k_hop=3, val_path=None):
    data_user = data[data['user_id'] == user].sort_values('time')
    u_time = data_user['time'].values
    u_seq = data_user['item_id'].values
    train_num, test_num = 0, 0
    # generate training data
    if len(u_seq) < 3:
        return train_num, test_num
    for j, t  in enumerate(u_time[1:-1]):
        # remove edges that are too new or old
        start_t = u_time[max(j - item_max_length, 0)]
        end_t = u_time[j + 1]
        sub_u_eid = (graph.edges['by'].data['time'] < end_t) & (graph.edges['by'].data['time'] >= start_t)
        sub_i_eid = (graph.edges['pby'].data['time'] < end_t) & (graph.edges['pby'].data['time'] >= start_t)
        sub_graph = dgl.edge_subgraph(graph, edges = {'by':sub_u_eid, 'pby':sub_i_eid}, relabel_nodes=False)
        u_temp, i_temp = torch.tensor([user]), torch.tensor([])
        his_user, his_item = torch.tensor([user]), torch.tensor([])
        edge_u, edge_i = [], []
        for hop in range(k_hop):
            if hop > 0:
                graph_u = select_topk(sub_graph, user_max_length, weight='time', nodes={'item': i_temp})  # item的邻居user
                u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype='pby')[0]), his_user)[-user_max_length:]
                his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
                edge_u.append(graph_u.edges['pby'].data[dgl.NID])
            graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user': u_temp})
            i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype='by')[0]), his_item)
            his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
            edge_i.append(graph_i.edges['by'].data[dgl.NID])
        all_edge_u = torch.unique(torch.cat(edge_u))
        all_edge_i = torch.unique(torch.cat(edge_i))
        fin_graph = dgl.edge_subgraph(sub_graph, edges={'by':all_edge_i,'pby':all_edge_u})
        target = u_seq[j+1]
        last_item = u_seq[j]
        u_alis = torch.where(fin_graph.nodes['user'].data['user_id']==user)[0]
        last_alis = torch.where(fin_graph.nodes['item'].data['item_id']==last_item)[0]
       
        # save graphs
        rel_path = '/' + str(user) + '/'+ str(user) + '_' + str(j) + '.bin'
        labels = {'user': torch.tensor([user]), 'target': torch.tensor([target]), 
                  'u_alis': u_alis, 'last_alis': last_alis}
        if j < len(u_time) - 3: # reserve last point for testing
            save_graphs(train_path + rel_path, fin_graph, labels)
            if j == len(u_time) - 4:
                save_graphs(val_path + rel_path, fin_graph, labels)
            train_num += 1
        else:
            save_graphs(test_path + rel_path, fin_graph, labels)
            test_num += 1
    return train_num, test_num


def generate_data(data, graph, item_max_length, user_max_length, train_path, test_path, val_path, job=10, k_hop=3):
    print('start data generation:', datetime.datetime.now(), flush=True)
    user = data['user_id'].unique()
    generate_func = lambda u: generate_user(u, data, graph, item_max_length, user_max_length, 
                                            train_path, test_path, k_hop, val_path)
    a = Parallel(n_jobs=1)(delayed(generate_func)(u) for u in user)
    return tuple([sum(tup) for tup in zip(*a)])


