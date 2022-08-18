#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/1 3:58
# @Author : ZM7
# @File : utils.py
# @Software: PyCharm
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import dgl
from dgl.sampling import sample_neighbors
from copy import copy
import torch
import numpy as np
from torch.nn.functional import one_hot
from bisect import bisect

class StaticData(Dataset):
    def __init__(self, root_dir, sampling=False, user_batching=1):
        self.root = root_dir
        def load_data(data_path):
            data_dir = []
            filenames = os.listdir(data_path)
            filenames.sort()
            for file in filenames:
                data_dir.append(os.path.join(data_path, file))
            return data_dir
        dir_list = load_data(root_dir)
        self.graph_list = [dgl.load_graphs(dir_) for dir_ in dir_list]
        self.batch_list = [labels['users'].shape[0] for _, labels in self.graph_list]
        self.num_users = np.sum(self.batch_list)
        if not sampling:
            self.batch_list = [x // user_batching + (x % user_batching > 0) for x in self.batch_list]
        self.batch_list = np.cumsum(self.batch_list)
        self.past_batch_list = np.roll(self.batch_list, 1)
        self.past_batch_list[0] = 0
        self.size = self.batch_list[-1]
            
        self.sampling = sampling
        self.batch_size = user_batching

    def __getitem__(self, index):
        list_idx = bisect(self.batch_list, index)
        graphs, labels = self.graph_list[list_idx]
        past_users = self.past_batch_list[list_idx]
        if self.sampling:
            labels = {key: val[index - past_users, ...].unsqueeze(0) for key, val in labels.items()}
        else:
            idx = index - past_users
            labels = {key: val[idx:idx + self.batch_size].unsqueeze(0) for key, val in labels.items()}
        return graphs[0], labels

    def __len__(self):
        return self.size

def user_neg(data, item_num):
    all_item = range(item_num)
    u_item = data.groupby('user_id')['item_id']
    return u_item.apply(lambda x: np.setdiff1d(all_item, x))

def neg_generate(user, label, neg_num=100, data_neg=None, item_num=None):
    if item_num:
        neg_num = 0
        label = pad(label, item_num)
    label = label.numpy()   
    B, T = label.shape                          # number of batches and max targets per batch
    neg = np.zeros((B, neg_num), np.int32)
    idx_flags = (label == -1) * np.arange(T, 0, -1)
    first_neg = np.argmax(idx_flags, -1)        # first idx to be -1 along last dim
    has_neg = np.max(label == -1, -1)           # whether there is a -1
    for i, u in enumerate(user):
        if item_num:
            neg_choices = np.setdiff1d(np.arange(item_num), label[i, :first_neg[i]])
        elif data_neg:
            neg_choices = data_neg[u.item()]
        neg[i] = np.random.choice(neg_choices, neg_num, replace=False)
        if has_neg[i]:                          # replace any -1s in label with neg samples
            idx = first_neg[i].item()
            label[i, idx:] = np.random.choice(neg_choices, T - idx, replace=False)
    all_label = torch.tensor(np.concatenate([label, neg], axis=-1)).long() 
    first_neg = torch.tensor(first_neg)
    return all_label, first_neg

def multihot(label, num_target, item_num):
    T, _ = label.shape
    target = -torch.ones((T, item_num), dtype=torch.float)
    for t in range(T):
        target[t, :] = one_hot(label[t, :num_target[t]], num_classes=item_num).float().sum(-2)
    return target

def pad(label, item_num):
    B, I = label.shape
    return torch.cat([label, -torch.ones((B, item_num - I)).long()], -1)
    

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
    user = torch.where(user[:, None] == khop.nodes['user'].data[dgl.NID][None, :])[1]
    return khop, user
    
    

def get_collate(item_num, max_items, max_users, k_hop, 
                train=True, margin=False, sampling=False, multiplier=1):
    def collate(data):
        # gather data (and subsample graphs if necessary)
        user, graphs, label, num_target = [], [], [], []
        for graph, labels in data:
            u = labels['users'].long()
            if len(u.shape) > 1:
                u = u.squeeze(-1)
            if sampling:
                samples = [subsample(graph, u, max_items, max_users, k_hop) for _ in range(multiplier)]
                graph, u = zip(*samples)
                u = torch.cat(u)
            else:
                graph = [graph]
            user.append(u)
            graphs.extend(graph)
            label.append(labels['items'])
            num_target.append(labels['num_items'])
        # batch and move to torch
        user = torch.cat(user).long()
        graphs = dgl.batch(graphs)
        label = torch.cat(label).long()
        num_target = torch.cat(num_target).long()
        # multihot-encode target and sample remaining users
        if margin:
            target = pad(label, item_num)
        else:
            target = multihot(label, num_target, item_num)
        # duplicate users/targets if using multiple graph samples
        if sampling and multiplier != 1:
            user = user.repeat(multiplier)
            target = target.repeat(multiplier, 1)

        if not train:
            return graphs, user, target, label, num_target
        else:
            return graphs, user, target
    return collate

def eval_metric(top, label, num_pos, ats=[5, 10, 20]):
    recalls = {at: [] for at in ats}
    ndcgs = {at: ([], []) for at in ats}
    B, K = top.shape
    _, I = label.shape
    top = top[:, None, :]
    label = label[:, :, None]
    ranks = torch.arange(K)[None, None, :].cuda()
    cgs = 1. / torch.log2(ranks + 2)
    matches = (top == label)
    mask = (torch.arange(K)[None, None, :].cuda() < num_pos[:, None, None])
    for at in ats:
        cg = cgs[:, :, :at]
        match = matches[:, :, :at]
        m = mask[:, :, :at]
        recalls[at] = (match.sum((1, 2)) / num_pos).mean()
        ndcgs[at] = ((match * cg).sum((1, 2)) / (m * cg).sum((1, 2))).mean()
    recalls = {f'recall@{at}': val.cpu().numpy().item() for at, val in recalls.items()}
    ndcgs = {f'ndcg@{at}': val.cpu().numpy().item() for at, val in ndcgs.items()}
    return {**recalls, **ndcgs}

def mkdir_if_not_exist(file_name):
    import os
    import shutil

    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
        
def long_substr(data):
    def is_substr(find, data):
        if len(data) < 1 and len(find) < 1:
            return False
        for i in range(len(data)):
            if find not in data[i]:
                return False
        return True
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and is_substr(data[0][i:i+j], data):
                    substr = data[0][i:i+j]
    return substr

def get_topk_items(score, target, num_target, k, neg_num=None):
    _, top_item = torch.topk(score, k, -1)
    if neg_num is None:
        return top_item
    num_erase = score.shape[1] - num_target.max() - neg_num
    erase_prob = torch.max(torch.tensor(0.), 1. - target)
    erase_idx = torch.multinomial(erase_prob, num_erase)
    sample_score = torch.clone(score)
    for i in range(score.shape[0]):
        sample_score[i, erase_idx[i, :]] = score[i, :].min()
    _, sample_top_item = torch.topk(sample_score, k, -1)
    return top_item, sample_top_item