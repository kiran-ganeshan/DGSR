#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/15 7:30
# @Author : ZM7
# @File : new_data
# @Software: PyCharm

import datetime
import itertools
import os
import pickle

#from sentence-tranformers import SentenceTransformer
import dgl
import numpy as np
import pandas as pd
import torch

from utils import mkdir_if_not_exist


def get_dataframes(raw_path):
    csv_names = os.listdir(raw_path)
    csv_names.sort()
    from_ntypes, to_ntypes = zip(*[name.split('_') for name in csv_names])
    to_ntypes = tuple(name.split('.')[0] for name in to_ntypes)
    from_to = itertools.chain(zip(from_ntypes, to_ntypes), zip(to_ntypes, from_ntypes))
    csv_names = itertools.chain(csv_names, csv_names)
    etypes = [(f, f[0] + t[0], t) for f, t in from_to]
    csvs = [pd.read_csv(raw_path + '/' + file) for file in csv_names]
    #csvs = [csv if 'user_id' not in csv.columns else csv[csv['user_id'] <= 100] for csv in csvs]
    return etypes, csvs

def add_edges(srcntype, dstntype, etype, src_id, dst_id, data, graph_data=({}, {})):
    edges, edata = graph_data
    edges[(srcntype, etype, dstntype)] = (torch.tensor(src_id), torch.tensor(dst_id))
    edata[etype] = {key: torch.tensor(val).long() for key, val in data.items()}
    return edges, edata

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
    enroll = enroll.groupby(['time', 'users'])
    enroll = pd.DataFrame({'items': enroll['items'].apply(lambda x: list(x))})
    enroll['num_items'] = enroll['items'].apply(lambda x: len(x))
    max_num_items = enroll['num_items'].max()
    enroll['items'] = enroll.apply(lambda r: [r['items'] + (max_num_items - r['num_items']) * [-1]], axis=1)
    enroll['items'] = enroll['items'].apply(lambda lst: lst[0])
    enroll = enroll.reset_index().groupby('time')
    keys = ['users', 'items', 'num_items']
    enroll = enroll.apply(lambda r: pd.Series([list(r[k]) for k in keys], index=keys)) 
        
    perm_edges = {key: graph.edges[key].data['time'] < 0 for key in graph.etypes}
    past_users = dgl.edge_subgraph(graph, perm_edges).nodes('user')
    for t, row in enroll.iterrows():
        # construct subgraph
        to_keep = lambda etype: torch.ones(graph.num_edges(etype)).long() if 'u' not in etype else (
             (graph.edges[etype].data['time'] < t) & 
             (graph.edges[etype].data['time'] >= t - max_lookback)
        )
        edges = {etype: to_keep(etype) for etype in graph.etypes}
        subgraph = dgl.edge_subgraph(graph, edges)
        new_users = np.setdiff1d(row['users'], subgraph.nodes['user'].data[dgl.NID])
        subgraph = dgl.add_nodes(subgraph, len(new_users), {dgl.NID: torch.tensor(new_users)}, 'user')
        # for etype in subgraph.etypes:
        #     subgraph.edges[etype].data['predict_time'] = torch.tensor([t]).repeat(subgraph.num_edges(etype))
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
        
        etypes, dfs = get_dataframes(raw_path)

        # graph
        graph = generate_graph(etypes, dfs)
        dgl.save_graphs(graph_path, graph)
        
        # metadata
        metadata = {'etypes': etypes, 'ntypes': graph.ntypes}
        metadata['num_nodes'] = {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
            
        # data
        print('start data generation:', datetime.datetime.now(), flush=True)
        enroll_df = [df for df, etype in zip(dfs, etypes) if etype[1] == 'ui'][0]
        path_args = (train_path, test_path, val_path)
        train_num, val_num, test_num = generate_data(enroll_df, graph, opt.max_lookback, 
                                                     opt.test_num, *path_args)
        
        # save metadata (last to indicate completion)
        with open(metadata_path, 'wb') as file:
            pickle.dump(metadata, file)
            
        print('The number of train set: ', train_num, flush=True)
        print('The number of val set: ', val_num, flush=True)
        print('The number of test set: ', test_num, flush=True)
        print('End preprocessing: ', datetime.datetime.now(), flush=True)
    return train_path, test_path, val_path, metadata['num_nodes'], metadata['etypes'], metadata['ntypes']

