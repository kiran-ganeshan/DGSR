#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 3:29
# @Author : ZM7
# @File : DGSR
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGSR(nn.Module):
    def __init__(self, user_num, item_num, input_dim, max_lookback, feat_drop=0.2, 
                 attn_drop=0.2, layer_num=3):
        super(DGSR, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = input_dim
        self.layer_num = layer_num

        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size)
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size)
        self.unified_map = nn.Linear(self.layer_num * self.hidden_size, 
                                     self.hidden_size, bias=False)
        self.layers = nn.ModuleList([DGSRLayers(self.hidden_size, max_lookback, feat_drop, attn_drop)
                                     for _ in range(self.layer_num)])
        self.reset_parameters()

    def forward(self, g, user, all_label=None, is_training=False):
        feat_dict = None
        user_layer = []
        g.nodes['user'].data['user_h'] = self.user_embedding(g.nodes['user'].data['user_id'].cuda())
        g.nodes['item'].data['item_h'] = self.item_embedding(g.nodes['item'].data['item_id'].cuda())
        if self.layer_num > 0:
            for conv in self.layers:
                feat_dict = conv(g, feat_dict)
                user_layer.append(graph_user(g, user, feat_dict['user']))
        unified_embedding = self.unified_map(torch.cat(user_layer, -1))
        score = torch.matmul(unified_embedding, self.item_embedding.weight.transpose(1, 0))
        score = torch.log_softmax(score, -1)
        if is_training:
            return score
        neg_embedding = self.item_embedding(all_label)
        score_neg = torch.matmul(unified_embedding.unsqueeze(-2), neg_embedding.transpose(-1, -2)).squeeze(-2)
        return score, score_neg

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)


class DGSRLayers(nn.Module):
    def __init__(self, hidden_size, max_lookback, feat_drop=0.2, attn_drop=0.2):
        super(DGSRLayers, self).__init__()
        self.hidden_size = hidden_size
        # self.agg_gate_u = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        # self.agg_gate_i = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        self.user_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        self.item_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        self.user_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.item_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)

        # attention+ attention mechanism
        # self.last_weight_u = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.last_weight_i = nn.Linear(hidden_size, hidden_size, bias=False)

        self.i_time_encoding = nn.Embedding(max_lookback, hidden_size)
        self.i_time_encoding_k = nn.Embedding(max_lookback, hidden_size)
        self.u_time_encoding = nn.Embedding(max_lookback, hidden_size)
        self.u_time_encoding_k = nn.Embedding(max_lookback, hidden_size)

    def user_update_function(self, user_now, user_old):
        return F.tanh(self.user_update(torch.cat([user_now, user_old], -1)))

    def item_update_function(self, item_now, item_old):
        return F.tanh(self.item_update(torch.cat([item_now, item_old], -1)))

    def forward(self, g, feat_dict=None):
        if feat_dict == None:
            user_ = g.nodes['user'].data['user_h']
            item_ = g.nodes['item'].data['item_h']
        else:
            user_ = feat_dict['user'].cuda()
            item_ = feat_dict['item'].cuda()
        g.nodes['user'].data['user_h'] = self.user_weight(self.feat_drop(user_))
        g.nodes['item'].data['item_h'] = self.item_weight(self.feat_drop(item_))
        g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func),
                            'pby': (self.item_message_func, self.item_reduce_func)}, 'sum')
        g.nodes['user'].data['user_h'] = self.user_update_function(g.nodes['user'].data['user_h'], user_)
        g.nodes['item'].data['item_h'] = self.item_update_function(g.nodes['item'].data['item_h'], item_)
        f_dict = {'user': g.nodes['user'].data['user_h'], 'item': g.nodes['item'].data['item_h']}
        return f_dict

    def item_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['user_h'] = edges.src['user_h']
        dic['item_h'] = edges.dst['item_h']
        return dic

    def item_reduce_func(self, nodes):
        order = nodes.mailbox['time']
        re_order = order.max() - order
        # order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        # re_order = nodes.mailbox['time'].shape[1] - order - 1

        e_ij = torch.sum((self.i_time_encoding(re_order) + nodes.mailbox['user_h']) * nodes.mailbox['item_h'], dim=2)\
                /torch.sqrt(torch.tensor(self.hidden_size).float())
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h_long = torch.sum(alpha * (nodes.mailbox['user_h'] + self.i_time_encoding_k(re_order)), dim=1)

        # length = nodes.mailbox['item_h'].shape[0]
        # last = torch.argmax(nodes.mailbox['time'], 1)
        # last_em = nodes.mailbox['user_h'][torch.arange(length), last, :].unsqueeze(1)
        # e_ij1 = torch.sum(last_em * nodes.mailbox['user_h'], dim=2) / torch.sqrt(
        #     torch.tensor(self.hidden_size).float())
        # alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
        # if len(alpha1.shape) == 2:
        #     alpha1 = alpha1.unsqueeze(2)
        # h_short = torch.sum(alpha1 * nodes.mailbox['user_h'], dim=1)

        # item_h = self.agg_gate_i(torch.cat([h_long, h_short], -1))
        item_h = h_long
        return {'item_h': item_h}

    def user_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['item_h'] = edges.src['item_h']
        dic['user_h'] = edges.dst['user_h']
        return dic

    def user_reduce_func(self, nodes):
        order = nodes.mailbox['time']
        re_order = order.max() - order
        # order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1),1)
        # re_order = nodes.mailbox['time'].shape[1] - order - 1

        e_ij = torch.sum((self.u_time_encoding(re_order) + nodes.mailbox['item_h']) *nodes.mailbox['user_h'],
                            dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h_long = torch.sum(alpha * (nodes.mailbox['item_h'] + self.u_time_encoding_k(re_order)), dim=1)

        # length = nodes.mailbox['user_h'].shape[0]
        # last = torch.argmax(nodes.mailbox['time'], 1)
        # last_em = nodes.mailbox['item_h'][torch.arange(length), last, :].unsqueeze(1)
        # e_ij1 = torch.sum(last_em * nodes.mailbox['item_h'], dim=2)/torch.sqrt(torch.tensor(self.hidden_size).float())
        # alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
        # if len(alpha1.shape) == 2:
        #     alpha1 = alpha1.unsqueeze(2)
        # h_short = torch.sum(alpha1 * nodes.mailbox['item_h'], dim=1)

        # user_h = self.agg_gate_u(torch.cat([h_long, h_short], -1))
        user_h = h_long
        return {'user_h': user_h}


def graph_user(bg, user_index, user_feats):
    b_user_size = bg.batch_num_nodes('user')
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1)
    tmp[0] = 0
    new_user_index = tmp + user_index
    return user_feats[new_user_index]


# def graph_item(bg, last_index, item_feats):
#     b_item_size = bg.batch_num_nodes('item')
#     tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
#     tmp[0] = 0
#     new_item_index = tmp + last_index
#     return item_feats[new_item_index]