#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/15 7:30
# @Author : ZM7
# @File : new_data
# @Software: PyCharm

import dgl
import numpy as np
import datetime
from dgl.sampling import select_topk
import torch
import os
from dgl import save_graphs
from joblib import Parallel, delayed

# Replace dataframe column values with relative order of values
def order(data, key):
    user_ids = np.unique(data[key].values)
    order = {k: v for k, v in zip(user_ids, range(len(user_ids)))}
    reverse = {v: k for k, v in order.items()}
    data[key] = data[key].apply(lambda id: order[id])
    return data, reverse

def preprocess_data(data):
    data, u_reverse = order(data, 'user_id')
    data, i_reverse = order(data, 'item_id')
    data = data.groupby('user_id').sort_values('time').reset_index(drop=True)
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


def generate_user(user, data, graph, item_num, item_max_length, user_max_length, 
                  train_path, test_path, k_hop=3, val_path=None, one_hot_labels=False):
    data_user = data[data['user_id'] == user]#.sort_values('time')
    u_time = data_user['time'].values
    u_seq = data_user['item_id'].values
    
    # basket-ize same-time items
    prev_time = -1
    u_basket_time = np.unique(u_time)       # basket-ized times
    u_basket_seq = []                       # basket-ized items
    u_basket_count = []                     # basket sizes
    for time, item in zip(u_time, u_seq):
        if time != prev_time:
            u_basket_seq.append([])
            u_basket_count.append(0)
            prev_time = time
        u_basket_seq[-1].append(item)
        u_basket_count[-1] += 1
    
    train_num, test_num = 0, 0
    if len(u_seq) < 3:                      # if not enough data for a training 
        return train_num, test_num          # example, ignore this user
    for j, t in enumerate(u_basket_time[1:]):
        # set target and most recent baskets
        target = u_basket_seq[j]
        last_basket = u_basket_seq[j - 1]
        # if not one_hot_labels:              # if we use BCELoss, create multi-hot target
        target += [-1] * (item_num - len(target))
        target = torch.tensor(target).long()
        # else:                               # if we use multiclass hinge, pad with -1
        #     target = torch.tensor(target)
        #     target = one_hot(target, num_classes=item_num).sum(0).float()
        
        # remove future edges
        edges = {key: graph.edges[key].data['time'] < t for key in ['by', 'pby']}
        sub_graph = dgl.edge_subgraph(graph, edges=edges, relabel_nodes=False)
        
        # extract k-hop subgraph
        u_temp, i_temp = torch.tensor([user]), torch.tensor([])
        his_user, his_item = torch.tensor([user]), torch.tensor([])
        edge_u, edge_i = [], []
        for hop in range(k_hop):
            if hop > 0:
                graph_u = select_topk(sub_graph, user_max_length, weight='time', nodes={'item': i_temp}) 
                u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype='pby')[0]), his_user)
                if user_max_length > 0:
                    u_temp = u_temp[-user_max_length:]
                his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
                edge_u.append(graph_u.edges['pby'].data[dgl.NID])
            graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user': u_temp})
            i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype='by')[0]), his_item)
            if item_max_length > 0:
                i_temp = i_temp[-item_max_length:]
            his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
            edge_i.append(graph_i.edges['by'].data[dgl.NID])
        all_edge_u = torch.unique(torch.cat(edge_u))
        all_edge_i = torch.unique(torch.cat(edge_i))
        fin_graph = dgl.edge_subgraph(sub_graph, edges={'by':all_edge_i,'pby':all_edge_u})
        
        u_alis = torch.where(fin_graph.nodes['user'].data['user_id']==user)[0]
        all_items = fin_graph.nodes['item'].data['item_id']
        last_alis = [item for item in all_items if item in last_basket]
        last_alis = torch.tensor(last_alis + [-1] * (item_num - len(last_alis)))
       
        # save graphs
        rel_path = '/' + str(user) + '/'+ str(user) + '_' + str(j) + '.bin'
        labels = {'target': target.unsqueeze(0), 'u_alis': u_alis, 'last_alis': last_alis}
        if j < len(u_time) - 3: # reserve last point for testing
            save_graphs(train_path + rel_path, fin_graph, labels)
            if j == len(u_time) - 4:
                save_graphs(val_path + rel_path, fin_graph, labels)
            train_num += 1
        else:
            save_graphs(test_path + rel_path, fin_graph, labels)
            test_num += 1
    return train_num, test_num


def generate_data(data, graph, item_num, item_max_length, user_max_length, 
                  train_path, test_path, val_path, job=10, k_hop=3, use_hinge=False):
    print('start data generation:', datetime.datetime.now(), flush=True)
    user = data['user_id'].unique()
    generate_func = lambda u: generate_user(u, data, graph, item_num, item_max_length, user_max_length, 
                                            train_path, test_path, k_hop, val_path, not use_hinge)
    a = Parallel(n_jobs=1)(delayed(generate_func)(u) for u in user)
    return tuple([sum(tup) for tup in zip(*a)])


