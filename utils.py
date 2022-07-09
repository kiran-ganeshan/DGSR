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
    label = label.numpy()
    neg = np.zeros((len(user), neg_num), np.int32)
    idx_flags = (label == -1) * np.arange(label.shape[1], 0, -1)
    first_idx = np.argmax(idx_flags, -1)        # first idx to be -1 along last dim
    has_idx = np.max(label == -1, -1)           # whether there is a -1
    for i, u in enumerate(user):
        neg_choices = data_neg[u.item()]
        neg[i] = np.random.choice(neg_choices, neg_num)
        if has_idx[i]:                          # replace any -1s in label with neg samples
            idx = first_idx[i].item()
            label[i, idx:] = np.random.choice(neg_choices, label.shape[1] - idx)
    all_label = torch.tensor(np.concatenate([label, neg], axis=-1)).long()
    return all_label, first_idx


class StaticData(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.dir_list = load_data(root_dir)
        self.size = len(self.dir_list)

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        return dgl.load_graphs(dir_)

    def __len__(self):
        return self.size

def get_collate(use_hinge):
    def collate(data):
        user = []
        graph = []
        label = []
        last = []
        for graphs, labels in data:
            user.append(labels['user'])
            graph.extend(graphs)
            label.append(labels['target'])
            last.append(labels['last'])
        user = torch.tensor(user).long()
        graph = dgl.batch(graph)
        label = torch.cat(label)
        if not use_hinge:
            label = label.float()
        last = torch.cat(last)
        return user, graph, label, last
    return collate

def load_data(data_path):
    data_dir = []
    dir_list = os.listdir(data_path)
    dir_list.sort()
    for name in dir_list:
        folder = os.path.join(data_path, name)
        for file in os.listdir(folder):
            data_dir.append(os.path.join(folder, file))
    return data_dir

def get_collate_test(use_hinge, item_num, data_neg):
    collate = get_collate(use_hinge)
    def collate_test(data):
        # generate negative samples
        user, graph, label, last_item = collate(data)
        all_label, neg_idx = neg_generate(user, label, data_neg)
        if not use_hinge:
            label = one_hot(label.long(), num_classes=item_num).sum(-2).float()
        return user, graph, label, last_item, all_label, neg_idx
    return collate_test

def eval_metric(all_scores, neg_idxs, ats=[5, 10, 20]):
    recalls = {at: [] for at in ats}
    ndcgs = {at: ([], []) for at in ats}
    for scores, neg_idx in zip(all_scores, neg_idxs):
        prediction = (-scores).argsort(1).argsort(1)
        for i, ranks in enumerate(prediction):
            for rank in ranks[:neg_idx[i]]:
                for at in ats:
                    if rank < at:
                        ndcgs[at][0].append(1 / np.log2(rank + 2))
                        recalls[at].append(1)
                    else:
                        ndcgs[at][0].append(0)
                        recalls[at].append(0)
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