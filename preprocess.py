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

# Replace dataframe column values with relative order of values
def remap_values(data, key):
    user_ids = np.unique(data[key].values)
    order = {k: v for k, v in zip(user_ids, range(len(user_ids)))}
    reverse = {v: k for k, v in order.items()}
    data = map_values(data, key, order)
    return data, order, reverse

def map_values(data, key, fwd):
    data[key] = data[key].apply(lambda id: fwd[id])
    return data

def relabel_data(enroll, student, course):
    enroll, u_fwd, u_rev = remap_values(enroll, 'user_id')
    enroll, i_fwd, i_rev = remap_values(enroll, 'item_id')
    student, _, m_rev = remap_values(student, 'major_id')
    student, d_fwd, d_rev = remap_values(student, 'dept_id')
    student = map_values(student, 'user_id', u_fwd)
    course = map_values(course, 'item_id', i_fwd)
    course = map_values(course, 'dept_id', d_fwd)
    course = map_values(course, 'prereq_id', i_fwd)
    return enroll, student, course, u_rev, i_rev, m_rev, d_rev

def add_edges(srcntype, dstntype, fwdetype, revetype, src, dst, time, graph_data=({}, {})):
    edata, times = graph_data
    edata[(srcntype, fwdetype, dstntype)] = (torch.tensor(src), torch.tensor(dst))
    edata[(dstntype, revetype, srcntype)] = (torch.tensor(dst), torch.tensor(src))
    times[fwdetype] = torch.tensor(time).long()
    times[revetype] = torch.tensor(time).long()
    return edata, times

def generate_graph(enroll, student, course, graph_repr=True):
    user = enroll['user_id'].values
    item = enroll['item_id'].values
    time = enroll['time'].values
    maj = student['major_id'].values
    maj_u = student['user_id'].values
    maj_dept = student['dept_id'].values
    prereq = course['prereq_id'].values
    prereq_i = course['item_id'].values
    
    graph_data = add_edges('user', 'item', 'sc', 'cs', user, item, time)
    graph_data = add_edges('item', 'item', 'cp', 'pc', prereq_i, prereq, data=graph_data)
    graph_data = add_edges('user', 'major', 'sm', 'ms', maj_u, maj, data=graph_data)
    graph_data = add_edges('major', 'dept', 'md', 'dm', maj, maj_dept, data=graph_data)
        
    # course = pd.merge(pd.DataFrame({'item_id': np.unique(item)}), course, 'left', 'item_id')
    # student = pd.merge(pd.DataFrame({'user_id': np.unique(user)}), student, 'left', 'user_id')
    
    edata, times = graph_data
    graph = dgl.heterograph(edata)
    for etype, time in times:
        graph.edges[etype].data['time'] = time
    graph.nodes['user'].data['user_id'] = torch.tensor(np.unique(user)).long()
    graph.nodes['item'].data['item_id'] = torch.tensor(np.unique(item)).long()
    #graph.nodes['item'].data['desc_h'] = SentenceTransformer(list(course['item_desc']))
    graph.nodes['major'].data['major_id'] = torch.tensor(np.unique(maj)).long()
    graph.nodes['dept'].data['dept_id'] = torch.tensor(np.unique(maj_dept)).long()
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
    if os.path.exists(metadata_path):
        print("skipped preprocessing", flush=True)
        with open(metadata_path, 'rb') as file:
            metadata = pickle.load(file)
            user_num = metadata['user_num']
            item_num = metadata['item_num']
            etypes = metadata['etypes']
        return train_path, test_path, val_path, user_num, item_num, etypes
    print('start preprocessing:', datetime.datetime.now(), flush=True)
    for path in [data_path, train_path, test_path, val_path]:
        if path:
            mkdir_if_not_exist(path)
    enroll = pd.read_csv('./data/' + opt.enroll + '.csv')
    course = pd.read_csv('./data/' + opt.course + '.csv')
    student = pd.read_csv('./data/' + opt.student + '.csv')
    
    # refine node indices
    enroll, u_rev, i_rev = relabel_data(enroll, student, course)
    
    # metadata
    metadata = {key + '_num': len(enroll[key + '_id'].unique()) for key in ['user', 'item']}
    metadata = {**metadata, 'user_rev': u_rev, 'item_rev': i_rev}
    
    # graph
    if not os.path.exists(graph_path):
        graph = generate_graph(enroll)
        dgl.save_graphs(graph_path, graph)
    else:
        graph = dgl.load_graphs(graph_path)[0][0]
    metadata = {**metadata, 'etypes': graph.canonical_etypes}
        
    # data
    print('start data generation:', datetime.datetime.now(), flush=True)
    path_args = (train_path, test_path, val_path)
    train_num, val_num, test_num = generate_data(enroll, graph, opt.max_lookback, opt.item_num, *path_args)
    
    # save metadata (last to indicate completion)
    with open(metadata_path, 'wb') as file:
        pickle.dump(metadata, file)
    user_num = metadata['user_num']
    item_num = metadata['item_num']
    etypes = metadata['etypes']
        
    print('The number of train set: ', train_num // opt.batch_size, flush=True)
    print('The number of val set: ', val_num // opt.batch_size, flush=True)
    print('The number of test set: ', test_num // opt.batch_size, flush=True)
    print('End preprocessing: ', datetime.datetime.now(), flush=True)
    return train_path, test_path, val_path, user_num, item_num, etypes

