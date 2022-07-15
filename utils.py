#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/1 3:58
# @Author : ZM7
# @File : utils.py
# @Software: PyCharm
import os
from torch.utils.data import Dataset
import dgl
import torch
import numpy as np
from torch.nn.functional import one_hot

class StaticData(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        def load_data(data_path):
            data_dir = []
            filenames = os.listdir(data_path)
            filenames.sort()
            for file in filenames:
                data_dir.append(os.path.join(data_path, file))
            return data_dir
        self.dir_list = load_data(root_dir)
        self.size = len(self.dir_list)

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        return dgl.load_graphs(dir_)

    def __len__(self):
        return self.size

def user_neg(data, item_num):
    all_item = range(item_num)
    u_item = data.groupby('user_id')['item_id']
    return u_item.apply(lambda x: np.setdiff1d(all_item, x))

def neg_generate(user, label, data_neg, neg_num=100):
    label = label.numpy()
    neg = np.zeros(user.shape + (neg_num,), np.int32)
    idx_flags = (label == -1) * np.arange(label.shape[-1], 0, -1)
    first_idx = np.argmax(idx_flags, -1)        # first idx to be -1 along last dim
    has_idx = np.max(label == -1, -1)           # whether there is a -1
    for i, timeslice in enumerate(user):
        for j, u in enumerate(timeslice):
            neg_choices = data_neg[u.item()]
            neg[i, j] = np.random.choice(neg_choices, neg_num)
            if has_idx[i, j]:                          # replace any -1s in label with neg samples
                idx = first_idx[i, j].item()
                label[i, j, idx:] = np.random.choice(neg_choices, label.shape[-1] - idx)
    all_label = torch.tensor(np.concatenate([label, neg], axis=-1)).long()
    return all_label, first_idx

def sample_users(user, label, target, num_user, num_target, n_user):
    T, U, I = target.shape
    _, _, L = label.shape
    num_extra = n_user - U
    assert num_extra > 0, f"max number of users is {U} but n_user is only {n_user}"
    extra_user = torch.empty((T, num_extra)).long()
    extra_target = torch.empty((T, num_extra, I)).float()
    extra_num_target = torch.empty((T, num_extra)).long()
    extra_label = torch.empty((T, num_extra, L)).long()
    for t, nu in enumerate(num_user):
        nu = nu.item()
        samples = np.random.choice(nu, user.shape[1] - nu)
        user[t, nu:] = user[t, samples]
        target[t, nu:, :] = target[t, samples, :]
        num_target[t, nu:] = num_target[t, samples]
        label[t, nu:] = label[t, samples]
        samples = np.random.choice(nu, num_extra)
        extra_user[t, :] = user[t, samples]
        extra_target[t, :, :] = target[t, samples, :]
        extra_num_target[t, :] = num_target[t, samples]
        extra_label[t, :, :] = label[t, samples, :]
    user = torch.cat([user, extra_user], 1)
    target = torch.cat([target, extra_target], 1)
    num_target = torch.cat([num_target, extra_num_target], 1)
    label = torch.cat([label, extra_label], 1)
    return user, label, target, num_target

def multihot(label, num_user, num_target, item_num):
    T, U, _ = label.shape
    target = -torch.ones((T, U, item_num), dtype=torch.float)
    for t, nu in enumerate(num_user):
        for u in range(nu):
            target[t, u, :] = one_hot(label[t, u, :num_target[t, u]].long(), num_classes=item_num).sum(-2).float()
    return target

def get_collate(item_num, n_user, device, data_neg=None):
    def collate(data):
        # gather data
        user, num_user, graph, label, num_target = [], [], [], [], []
        for graphs, labels in data:
            user.append(labels['users'])
            num_user.append(labels['num_users'])
            graph.extend(graphs)
            label.append(labels['items'])
            num_target.append(labels['num_items'])
        # batch and move to torch
        user = torch.cat(user).long()
        graph = dgl.batch(graph)
        label = torch.cat(label).float()
        num_target = torch.cat(num_target).long()
        num_user = torch.tensor(num_user).long()
        # multihot-encode target and sample remaining users
        target = multihot(label, num_user, num_target, item_num)
        user, label, target, num_target = sample_users(user, label, target, num_user, num_target, n_user)
        # generate negatives if required
        tup = () if data_neg is None else neg_generate(user, label, data_neg)
        # move to cuda
        #graph = graph.to(device)
        #user = user.to(device)
        #new_target = new_target.to(device)
        return graph, user, target, *tup
    return collate

def eval_metric(all_scores, neg_idxs, ats=[5, 10, 20]):
    recalls = {at: [] for at in ats}
    ndcgs = {at: ([], []) for at in ats}
    all_scores = np.concatenate(all_scores)
    all_scores = all_scores.reshape((-1, all_scores.shape[-1]))
    neg_idxs = np.concatenate(neg_idxs).reshape(-1)
    prediction = (-all_scores).argsort(-1).argsort(-1)
    for ranks, idx in zip(prediction, neg_idxs):
        for i, rank in enumerate(ranks[:idx]):
            for at in ats:
                if rank < at:
                    ndcgs[at][0].append(1 / np.log2(rank + 2))
                    recalls[at].append(1)
                else:
                    ndcgs[at][0].append(0)
                    recalls[at].append(0)
                if i < at:
                    ndcgs[at][1].append(1 / np.log2(i + 2))
    recalls = {f'recall@{at}': np.mean(lst) for at, lst in recalls.items()}
    ndcgs = {f'ndcg@{at}': np.sum(num) / np.sum(denom) for (at, (num, denom)) in ndcgs.items()}
    return {**recalls, **ndcgs}

def mkdir_if_not_exist(file_name):
    import os
    import shutil

    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)