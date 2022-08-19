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

def generate_data(data, graph, max_lookback, train_path, test_path, val_path, test_num):
    # get cutoff time
    times = np.sort(np.unique(data['time'].values))
    t_cutoff = times[-test_num - 1]
    train_num, test_num, val_num = 0, 0, 0
    # process dataframe
    data = data.rename(columns={'user_id': 'users', 'item_id': 'items'})
    user_num = data['users'].max() + 1
    data = data.groupby(['time', 'users'])
    data = pd.DataFrame({'items': data['items'].apply(lambda x: list(x))})
    data['num_items'] = data['items'].apply(lambda x: len(x))
    max_num_items = data['num_items'].max()
    data['items'] = data.apply(lambda r: [r['items'] + (max_num_items - r['num_items']) * [-1]], axis=1)
    data['items'] = data['items'].apply(lambda lst: lst[0])
    data = data.reset_index().groupby('time')
    keys = ['users', 'items', 'num_items']
    data = data.apply(lambda r: pd.Series([list(r[k]) for k in keys], index=keys))
    # record data          
    user_counts = [torch.zeros((user_num,))]
    for t, row in data.iterrows():
        # process user counts
        curr_users = torch.tensor(row['users'], dtype=torch.long)
        curr_count = torch.zeros((user_num,))
        curr_count[curr_users] = 1
        recent_count = user_counts[-1] - (user_counts[t - max_lookback] if t >= max_lookback else 0)
        user_counts.append(user_counts[-1] + curr_count)
        # construct subgraph
        new_users = torch.where(curr_count - (recent_count > 0).to(torch.int) > 0)[0]
        edges = {key: (graph.edges[key].data['time'] < t) & 
                      (graph.edges[key].data['time'] >= t - max_lookback) 
                      for key in graph.etypes}
        subgraph = dgl.edge_subgraph(graph, edges)
        subgraph = dgl.add_nodes(subgraph, len(new_users), {dgl.NID: new_users}, 'user')
        for etype in subgraph.etypes:
            subgraph.edges[etype].data['predict_time'] = torch.tensor([t]).repeat(subgraph.num_edges(etype))
        # construct labels and alias users to subgraph IDs
        rel_path = '/' + str(t) + '.bin'
        keys = ['items', 'users', 'num_items']
        labels = {key: torch.tensor(row[key]).long() for key in keys}
        alias_match = (labels['users'][:, None] == subgraph.nodes['user'].data[dgl.NID][None, :])
        labels['users'] = torch.where(alias_match)[1].squeeze()
        # save subgraph and labels
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
    


