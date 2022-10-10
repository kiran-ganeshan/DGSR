#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/15 7:30
# @Author : ZM7
# @File : new_data
# @Software: PyCharm

#from sentence-tranformers import SentenceTransformer
import dgl
import numpy as np
import pandas as pd
import torch
from utils import mkdir_if_not_exist, user_neg
import datetime
import os
import pickle
import itertools

def add_edges(srcntype, dstntype, etype, src_id, dst_id, data, graph_data=({}, {}, {})):
    edges, edata, ndata = graph_data
    edges[(srcntype, etype, dstntype)] = (torch.tensor(src_id), torch.tensor(dst_id))
    edata[etype] = {key: torch.tensor(val).long() for key, val in data.items()}
    return edges, edata, ndata

def generate_graph(etypes, csvs):
    for (srcntype, etype, dstntype), csv in zip(etypes, csvs):
        src_id, dst_id = csv[f'{srcntype}_id'].values, csv[f'{dstntype}_id'].values
        edata = {key: csv[key].values for key in ['time', 'semester']}
        graph_data = add_edges(srcntype, dstntype, etype, src_id, dst_id, edata)
    edges, edata = graph_data
    graph = dgl.heterograph(edges)
    for etype, data in edata.items():
        for key, val in data.items():
            graph.edges[etype].data[key] = val
    return graph

def generate_data(enroll, graph, max_lookback, test_num, train_path, test_path, val_path):
    # get cutoff time
    times = np.sort(np.unique(enroll['time'].values))
    t_cutoff = times[-test_num - 1]
    train_num, test_num, val_num = 0, 0, 0
    # process dataframe
    enroll = enroll.rename(columns={'user_id': 'users', 'item_id': 'items'})
    user_num = enroll['users'].max() + 1
    enroll = enroll.groupby(['time', 'users'])
    enroll = pd.DataFrame({'items': enroll['items'].apply(lambda x: list(x))})
    enroll['num_items'] = enroll['items'].apply(lambda x: len(x))
    max_num_items = enroll['num_items'].max()
    enroll['items'] = enroll.apply(lambda r: [r['items'] + (max_num_items - r['num_items']) * [-1]], axis=1)
    enroll['items'] = enroll['items'].apply(lambda lst: lst[0])
    enroll = enroll.reset_index().groupby('time')
    keys = ['users', 'items', 'num_items']
    enroll = enroll.apply(lambda r: pd.Series([list(r[k]) for k in keys], index=keys))
    # record data          
    user_counts = [torch.zeros((user_num,))]
    for t, row in enroll.iterrows():
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
    
def preprocess(opt, data_path):
    train_path = data_path + 'train/'
    test_path = data_path + 'test/'
    val_path = data_path + 'val/' if opt.val else None
    graph_path = data_path + 'graph'
    metadata_path = data_path + 'meta'
    raw_path = './data/' + opt.data
    if os.path.exists(metadata_path):
        print("skipped preprocessing", flush=True)
        with open(metadata_path, 'rb') as file:
            metadata = pickle.load(file)
    else:
        print('start preprocessing:', datetime.datetime.now(), flush=True)
        assert os.path.exists(raw_path), f"raw data folder {raw_path} does not exist"
        for path in [data_path, train_path, test_path, val_path]:
            if path:
                mkdir_if_not_exist(path)
        
        csv_names = os.listdir(raw_path)
        csv_names.sort()
        from_ntypes, to_ntypes = zip(*[name.split('_') for name in csv_names])
        to_ntypes = [name.split('.')[0] for name in to_ntypes]
        ntypes = np.unique(from_ntypes + to_ntypes).tolist()
        from_to = itertools.chain(zip(from_ntypes, to_ntypes), zip(to_ntypes, from_ntypes))
        etypes = [(f, f[0] + t[0], t) for f, t in from_to]
        csvs = [pd.read_csv(data_path + file) for file in csv_names]
        
        # metadata
        metadata = {'etypes': etypes, 'ntypes': ntypes}
        metadata['num_nodes'] = {ntype: graph.num_nodes(ntype) for ntype in ntypes}
        
        # graph
        graph = generate_graph(etypes, csvs)
        dgl.save_graphs(graph_path, graph)
            
        # data
        print('start data generation:', datetime.datetime.now(), flush=True)
        enroll_csv = [csv for csv, etype in zip(csvs, etypes) if etype[1] == 'ui'][0]
        path_args = (train_path, test_path, val_path)
        train_num, val_num, test_num = generate_data(enroll_csv, graph, opt.max_lookback, 
                                                     metadata['num_nodes']['item'], *path_args)
        
        # save metadata (last to indicate completion)
        with open(metadata_path, 'wb') as file:
            pickle.dump(metadata, file)
            
        print('The number of train set: ', train_num // opt.batch_size, flush=True)
        print('The number of val set: ', val_num // opt.batch_size, flush=True)
        print('The number of test set: ', test_num // opt.batch_size, flush=True)
        print('End preprocessing: ', datetime.datetime.now(), flush=True)
    return train_path, test_path, val_path, metadata['num_nodes'], metadata['etypes'], metadata['ntypes']

