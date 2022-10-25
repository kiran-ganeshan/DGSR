#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/1 3:58
# @Author : ZM7
# @File : utils.py
# @Software: PyCharm
import os
from bisect import bisect_right as bisect

import dgl
import numpy as np
import torch
from dgl.sampling import sample_neighbors
from torch.nn.functional import one_hot
from torch.utils.data import Dataset


class GraphData(Dataset):
    
    def __init__(self, root_dir):
        self.root = root_dir
        def load_data(data_path):
            data_dir = []
            filenames = os.listdir(data_path)
            filenames.sort()
            for file in filenames:
                data_dir.append(os.path.join(data_path, file))
            return data_dir
        dir_lst = load_data(root_dir)
        graph_lst, label_lst = zip(*[dgl.load_graphs(dir_) for dir_ in dir_lst])
        self.graph_lst = graph_lst
        self.label_lst = label_lst
        self.size = len(graph_lst)

    def __getitem__(self, index):
        graphs = self.graph_lst[index]
        labels = self.label_lst[index]
        return graphs[0], labels

    def __len__(self):
        return self.size
    
class SamplingGraphData(GraphData):
    
    def __init__(self, root_dir):
        super(SamplingGraphData, self).__init__(root_dir)
        self.num_user_lst = [labels['users'].shape[0] for labels in self.label_lst]
        self.num_user_lst = np.cumsum(self.num_user_lst)
        self.batch_idx_lst = np.roll(self.num_user_lst, 1)
        self.batch_idx_lst[0] = 0
        self.size = self.num_user_lst[-1]
        
    def __getitem__(self, index):
        list_idx = bisect(self.num_user_lst, index)
        graph, labels = super(SamplingGraphData, self).__getitem__(list_idx)
        batch_idx = self.batch_idx_lst[list_idx]
        labels = {key: val[index - batch_idx, ...].unsqueeze(0) for key, val in labels.items()}
        return graph, labels
    
    def __len__(self):
        return self.size

def multihot(label, num_target, item_num):
    T, _ = label.shape
    target = -torch.ones((T, item_num), dtype=torch.float)
    for t in range(T):
        target[t, :] = one_hot(label[t, :num_target[t]], num_classes=item_num).float().sum(-2)
    return target

def get_batched_user(user, graphs, batch_idx):
    past_nodes = torch.cumsum(graphs.batch_num_nodes('user'), 0)
    past_nodes = torch.roll(past_nodes, 1)
    past_nodes[0] = 0
    return user + past_nodes[batch_idx]

def pad(label, item_num):
    B, I = label.shape
    pad = -torch.ones((B, item_num - I)).long()
    return torch.cat([label, pad], -1)
    

empty = lambda: torch.tensor([]).long()
def empty_like(graph):
    edge_lst = {etype: (empty(), empty()) for etype in graph.canonical_etypes}
    num_nodes = {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
    return dgl.heterograph(edge_lst, num_nodes)

def subsample(graph, user, max_items, max_users, k):
    rev_etypes = {'by': 'pby', 'pby': 'by'}
    n_limit = {ntype: -1 for ntype in graph.ntypes}
    n_limit['user'] = max_users
    n_limit['item'] = max_items
    e_limit = {dir: {etype: -1 for etype in graph.etypes} for dir in ['in', 'out']}
    for outtype, etype, intype in graph.canonical_etypes:
            e_limit['in'][etype] = n_limit[intype]
            e_limit['out'][etype] = n_limit[outtype]
    nodes = {ntype: empty() for ntype in graph.ntypes}
    new_nodes = {ntype: empty() for ntype in graph.ntypes}
    nodes['user'] = user
    new_nodes['user'] = user
    # print({key: val.shape for key, val in new_nodes.items()}, flush=True)
    # print({key: val.shape for key, val in nodes.items()}, flush=True)
    # graph = copy(graph)
    # khop = empty_like(graph)
    edges = {etype: empty() for etype in graph.etypes}
    for _ in range(k):
        samples = dgl.merge([sample_neighbors(graph, new_nodes, e_limit[dir], dir) for dir in ['in', 'out']])
        for srcntype, fwdetype, dstntype in samples.canonical_etypes:
            revetype = rev_etypes[fwdetype]
            src, dst = samples.edges('uv', etype=fwdetype)
            fwd_eid = graph.edge_ids(src, dst, etype=fwdetype)
            edges[fwdetype] = torch.unique(torch.cat([edges[fwdetype], fwd_eid]))
            # graph.remove_edges(fwd_eid, fwdetype)
            rev_eid = graph.edge_ids(dst, src, etype=revetype)
            edges[revetype] = torch.unique(torch.cat([edges[revetype], rev_eid]))
            # graph.remove_edges(rev_eid, revetype)
            for ntype, node_lst in zip((srcntype, dstntype), (src, dst)):
                new_nodes[ntype] = torch.unique(torch.cat([node_lst, new_nodes[ntype]]))
        for ntype in graph.ntypes:
            new_nodes[ntype] = torch.tensor(np.setdiff1d(new_nodes[ntype], nodes[ntype]))
            nodes[ntype] = torch.unique(torch.cat([new_nodes[ntype], nodes[ntype]]))
    khop = dgl.edge_subgraph(graph, edges)
    if khop.num_nodes('user') == 0:
        khop = dgl.add_nodes(khop, 1, {dgl.NID: user}, 'user')
    user = torch.where(user[:, None] == khop.nodes['user'].data[dgl.NID][None, :])[1]
    return khop, user
    
def get_collate(item_num, max_items, max_users, k_hop, train=True, sampling=False):
    def collate(data):
        user, batch_idxs, graphs, label, num_target = [], [], [], [], []
        for i, (graph, labels) in enumerate(data):
            u = labels['users'].long()
            if sampling:
                graph, u = subsample(graph, u, max_items, max_users, k_hop)
                u = torch.cat(u)
            else:
                graph = [graph]
            user.append(u)
            graphs.extend(graph)
            label.append(labels['items'])
            num_target.append(labels['num_items'])
            batch_idxs.append(torch.full_like(u, i))
        user = torch.cat(user).long()
        batch_idx = torch.cat(batch_idxs).long()
        graphs = dgl.batch(graphs)
        label = torch.cat(label).long()
        num_target = torch.cat(num_target).long()
        user = get_batched_user(user, graphs, batch_idx)
        target = multihot(label, num_target, item_num)
        test_ex = () if train else (label, num_target) 
        return graphs, user, target, *test_ex
    return collate

def eval_metric(top, label, num_target, ats=[5, 10, 20]):
    _, K = top.shape
    top = top[:, None, :]
    label = label[:, :, None]
    num_pos = num_target[:, None]
    chunk_lens = [b - a for a, b in zip([0] + ats[:-1], ats)]
    match = (top == label).sum(1)
    def split_and_sum(x):
        chunks = torch.split(x, chunk_lens, -1)
        x = torch.stack([chunk.sum(-1) for chunk in chunks], -1)
        x = torch.cumsum(x, -1)
        return x
    def get_recalls():
        num_rel = split_and_sum(match)
        recall = (num_rel / num_pos).mean(0)
        return {f'recall@{at}': recall[i].item() for i, at in enumerate(ats)}
    def get_precisions():
        ranks = torch.arange(K)[None, :].cuda()
        cg = (1. / torch.log2(ranks + 2))
        mask = ranks < num_pos
        dcg = split_and_sum(match * cg)
        norm = split_and_sum(mask * cg)
        ndcg = (dcg / norm).mean(0)
        return {f'ndcg@{at}': ndcg[i].item() for i, at in enumerate(ats)}
    recalls = get_recalls()
    ndcgs = get_precisions()
    return {**recalls, **ndcgs}

def get_topk_items(score, target, num_target, k, neg_num=None):
    _, top_item = torch.topk(score, k, -1)
    if neg_num is None:
        return top_item
    num_erase = score.shape[1] - num_target.max() - neg_num
    erase_prob = torch.max(torch.tensor(0.), 1. - target)
    erase_idx = torch.multinomial(erase_prob, num_erase)
    sample_score = torch.clone(score)
    for i in range(score.shape[0]):
        sample_score[i, erase_idx[i, :]] = score[i, :].min() - 1.
    _, sample_top_item = torch.topk(sample_score, k, -1)
    return top_item, sample_top_item

def mkdir_if_not_exist(file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
        
def unchunk_list(lst):
    return [item for chunk in lst for item in chunk]
        
def chunk_list(lst, chunk_size, prefix=0, suffix=0):
    def _chunk(lst, chunk_size):
        end = len(lst) - suffix
        if prefix > 0:
            yield lst[:prefix]
        for i in range(prefix, end, chunk_size):
            yield lst[i:min(i + chunk_size, end)]
        if suffix > 0:
            yield lst[-suffix:]
    return list(_chunk(lst, chunk_size))
        
