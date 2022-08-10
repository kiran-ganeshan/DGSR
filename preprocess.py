#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/15 7:30
# @Author : ZM7
# @File : new_data
# @Software: PyCharm

from xml.dom.minicompat import NodeList
import dgl
import numpy as np
import pandas as pd
import torch
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
    return data, u_reverse, i_reverse

def generate_graph(data):
    user = data['user_id'].values
    item = data['item_id'].values
    time = data['time'].values
    
    graph_data = {('item','by','user'):(torch.tensor(item), torch.tensor(user)),
                  ('user','pby','item'):(torch.tensor(user), torch.tensor(item))}
    graph = dgl.heterograph(graph_data)
    graph.edges['by'].data['time'] = torch.tensor(time).long()
    graph.edges['pby'].data['time'] = torch.tensor(time).long()

    graph.nodes['user'].data['user_id'] = torch.tensor(np.unique(user)).long()
    graph.nodes['item'].data['item_id'] = torch.tensor(np.unique(item)).long()
    return graph

def generate(data, graph, max_lookback, t_cutoff, train_path, test_path, val_path):
    train_num, test_num, val_num = 0, 0, 0
    data = data.rename(columns={'user_id': 'users', 'item_id': 'items'})
    data = data.groupby(['time', 'users'])
    data = pd.DataFrame({'items': data['items'].apply(lambda x: list(x))})
    data['num_items'] = data['items'].apply(lambda x: len(x))
    max_num_items = data['num_items'].max()

    data['items'] = data.apply(lambda r: [r['items'] + (max_num_items - r['num_items']) * [-1]], axis=1)
    data['items'] = data['items'].apply(lambda lst: lst[0])
    data = data.reset_index().groupby('time')
    keys = ['users', 'items', 'num_items']
    data = data.apply(lambda r: pd.Series([list(r[k]) for k in keys], index=keys))
    # data['num_users'] = data['users'].apply(lambda x: len(x))
    # max_users = data['num_users'].max()
    # def pad(data, key, padding=-1):
    #     data[key] = data.apply(lambda r: [r[key] + (max_users - r['num_users']) * [padding]], axis=1)
    #     data[key] = data[key].apply(lambda lst: lst[0])
    #     return data
    # data = pad(data, 'users')
    # data = pad(data, 'items', padding=item_num * [-1])
    # data = pad(data, 'num_items')
    for t, row in data.iterrows():
        edges = {key: (graph.edges[key].data['time'] < t) & 
                      (graph.edges[key].data['time'] >= t - max_lookback) 
                      for key in graph.etypes}
        subgraph = dgl.edge_subgraph(graph, edges, relabel_nodes=False)
        rel_path = '/' + str(t) + '.bin'
        keys = ['items', 'users', 'num_items']
        labels = {key: torch.tensor(row[key]).long() for key in keys}
        labels = {**labels, 'time': torch.tensor([t]).long().repeat(labels['users'].shape[0])}
        if t == t_cutoff and val_path is not None:
            dgl.save_graphs(val_path + rel_path, subgraph, labels)
            val_num += 1
        elif t <= t_cutoff:
            dgl.save_graphs(train_path + rel_path, subgraph, labels)
            train_num += 1
        else:
            dgl.save_graphs(test_path + rel_path, subgraph, labels)
            test_num += 1
    return train_num, val_num, test_num
    


def generate_user(user, data, graph, max_lookback, t_cutoff, 
                  train_path, test_path, val_path, k_hop):
    data = data[data['user_id'] == user].sort_values('time')
    u_time = data['time'].values
    u_seq = data['item_id'].values
    
    # basket-ize same-time items
    prev_time = -1
    basket_time = np.unique(u_time)       # basket-ized times
    u_basket_seq = []                       # basket-ized items
    for time, item in zip(u_time, u_seq):
        if time != prev_time:
            u_basket_seq.append([])
            prev_time = time
        u_basket_seq[-1].append(item)
    
    train_num, val_num, test_num = 0, 0, 0
    if len(u_seq) < 2:                      # if not enough data for a training 
        return train_num, val_num, test_num # example, ignore this user
    for t, target in zip(basket_time[1:], u_basket_seq[1:]):
        # remove future edges and past edges older than max_lookback
        edges = {key: (graph.edges[key].data['time'] < t) & 
                      (graph.edges[key].data['time'] >= t - max_lookback) 
                      for key in graph.etypes}
        subgraph = dgl.edge_subgraph(graph, edges, relabel_nodes=False)
        
        # extract k-hop subgraph
        prev_num_nodes, num_nodes = 0, 1
        edges = {etype: [] for etype in subgraph.etypes}
        nodes = {ntype: torch.tensor([]).long() for ntype in subgraph.ntypes}
        nodes['user'] = torch.tensor([user]).long()
        new_nodes = nodes.copy()
        hop = 0
        while (k_hop < 0 or hop < k_hop) and num_nodes > prev_num_nodes:    # get nodes in k-hop subgraph
            prev_num_nodes = num_nodes
            for srctype, etype, dsttype in subgraph.canonical_etypes:
                for intype, outtype in [(srctype, dsttype), (dsttype, srctype)]:
                    func = subgraph.successors if intype == srctype else subgraph.predecessors
                    new_nodes[outtype] = [new_nodes[outtype]] + [func(i, etype) for i in nodes[intype]]
                    new_nodes[outtype] = torch.unique(torch.cat(new_nodes[outtype]))
            nodes = new_nodes.copy()
            num_nodes = sum([len(nodelst) for nodelst in nodes.values()])
            hop += 1
        for srctype, etype, dsttype in subgraph.canonical_etypes:              # get edges between nodes
            for src, dst, eid in zip(*[x.tolist() for x in subgraph.edges('all', etype=etype)]):
                if src in nodes[srctype] and dst in nodes[dsttype]:
                    edges[etype].append(eid)
        subgraph = dgl.edge_subgraph(subgraph, edges, relabel_nodes=False)
        
        # save graphs
        rel_path = '/' + str(user) + '_' + str(t) + '.bin'
        labels = {'items': [target], 'users': [user], 'num_items': [len(target)], 'num_users': 1}
        labels = {key: torch.tensor([val]).long() for key, val in labels.items()}
        labels = {**labels, 'time': torch.tensor([t]).long()}
        if t == t_cutoff and val_path is not None:
            save_graphs(val_path + rel_path, subgraph, labels)
            val_num += 1
        elif t <= t_cutoff:
            save_graphs(train_path + rel_path, subgraph, labels)
            train_num += 1
        else:
            save_graphs(test_path + rel_path, subgraph, labels)
            test_num += 1
    return train_num, val_num, test_num


def generate_data(data, graph, max_lookback, train_path, test_path, val_path, test_num, k_hop):
    times = np.sort(np.unique(data['time'].values))
    t_cutoff = times[-test_num - 1]
    return generate(data, graph, max_lookback, 
                    t_cutoff, train_path, test_path, val_path)
    


