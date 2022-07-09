#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/15 7:30
# @Author : ZM7
# @File : new_data
# @Software: PyCharm

from xml.dom.minicompat import NodeList
import dgl
import numpy as np
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


def generate_user(user, data, graph, item_num, max_lookback, t_cutoff, 
                  train_path, test_path, val_path, k_hop):
    data = data.sort_values('time')
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
    for j, t in enumerate(basket_time[1:]):
        # set target and most last basket
        target = u_basket_seq[j + 1]
        last_basket = u_basket_seq[j]
        
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
        
        # prune & pad target and last basket
        last_basket = [item for item in nodes['item'] if item in last_basket]
        last_basket += [-1] * (item_num - len(last_basket))
        target += [-1] * (item_num - len(target))
        
        # save graphs
        rel_path = '/' + str(user) + '/' + str(t) + '.bin'
        labels = {'target': target, 'user': user, 'last': last_basket, 'time': t}
        labels = {key: torch.tensor([val]).long() for key, val in labels.items()}
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


def generate_data(data, graph, item_num, max_lookback, train_path, test_path, val_path, 
                  job, k_hop, test_split=None, test_num=None):
    user = data['user_id'].unique()
    assert test_split or test_num
    times = np.sort(np.unique(data['time'].values))
    test_split = None if test_split is None else max(int(test_split * len(times)), 1)
    n_test_t = test_split or test_num
    t_cutoff = times[-n_test_t - 1]
    generate_func = lambda u: generate_user(u, data, graph, item_num, max_lookback, t_cutoff, 
                                            train_path, test_path, val_path, k_hop)
    a = Parallel(n_jobs=job)(delayed(generate_func)(u) for u in user)
    return tuple([sum(tup) for tup in zip(*a)])
    


