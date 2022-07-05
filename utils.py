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


def user_neg(data, item_num):
    all_item = range(item_num)
    u_item = data.groupby('user_id')['item_id']
    return u_item.apply(lambda x: np.setdiff1d(all_item, x))

def neg_generate(user, label, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    # find first idx along last dim of label to be -1
    first_idx = torch.argmax((label == -1) * torch.arange(label.shape[1], 0, -1), -1)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u.item()], neg_num)
        label[i, first_idx[i]:] = np.random.choice(data_neg[u.item()], label.shape[1] - first_idx[i])
    neg = torch.cat([label, torch.tensor(neg).long()], -1)
    return neg, first_idx


class StaticData(Dataset):
    def __init__(self, root_dir, loader):
        self.root = root_dir
        self.loader = loader
        self.dir_list = load_data(root_dir)
        self.size = len(self.dir_list)

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        data = self.loader(dir_)
        return data

    def __len__(self):
        return self.size


def collate(data):
    user_l = []
    graph = []
    label = []
    last_item = []
    for graphs, labels in data:
        user_l.append(labels['u_alis'])
        graph.append(graphs[0])
        label.append(labels['target'])
        last_item.append(labels['last_alis'])
    user_l = torch.tensor(user_l).long()
    graph = dgl.batch(graph)
    label = torch.cat(label)
    last_item = torch.cat(last_item).long()
    return user_l, graph, label, last_item


def load_data(data_path):
    data_dir = []
    dir_list = os.listdir(data_path)
    dir_list.sort()
    for name in dir_list:
        folder = os.path.join(data_path, name)
        for file in os.listdir(folder):
            data_dir.append(os.path.join(folder, file))
    return data_dir

def get_collate_test(use_hinge, item_num):
    def collate_test(data, user_neg):
        # generate negative samples
        user, graph, label, last_item = collate(data)
        all, neg_idx = neg_generate(user, label, user_neg)
        if not use_hinge:
            label = one_hot(label, num_classes=item_num).sum(-2).float()
        return user, graph, label, last_item, all, neg_idx
    return collate_test


def eval_metric(all_scores, neg_idxs, ats=[5, 10, 20]):
    recalls = {at: [] for at in ats}
    ndcgs = {at: [] for at in ats}
    for scores, neg_idx in zip(all_scores, neg_idxs):
        prediction = (-scores).argsort(1).argsort(1)
        for i, ranks in enumerate(prediction):
            for rank in ranks[:neg_idx[i]]:
                for at in ats:
                    if rank < at:
                        ndcgs[at].append(1 / np.log2(rank + 2))
                        recalls[at].append(1)
                    else:
                        ndcgs[at].append(0)
                        recalls[at].append(0)
    recalls = {'recall@' + at: np.mean(lst) for at, lst in recalls.items()}
    ndcgs = {'ndcg@' + at: np.mean(lst) for at, lst in ndcgs.items()}
    return {**recalls, **ndcgs}



def mkdir_if_not_exist(file_name):
    import os
    import shutil

    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)