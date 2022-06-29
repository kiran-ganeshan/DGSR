#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 3:29
# @Author : ZM7
# @File : DGSR
# @Software: PyCharm
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DGSR(nn.Module):
    def __init__(self, user_num, item_num, input_dim, item_max_length, user_max_length, feat_drop=0.2, attn_drop=0.2,
                 layer_num=3, time=True):
        super(DGSR, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = input_dim
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        self.layer_num = layer_num
        self.time = time

        # # long- and short-term encoder
        # self.user_long = user_long
        # self.item_long = item_long
        # self.user_short = user_short
        # self.item_short = item_short
        # # update function
        # self.user_update = user_update
        # self.item_update = item_update

        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size)
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size)
        self.unified_map = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size, bias=False)
        self.layers = nn.ModuleList([DGSRLayers(self.hidden_size, self.user_max_length, self.item_max_length, feat_drop, attn_drop)
                                     for _ in range(self.layer_num)])
        self.reset_parameters()

    def forward(self, g, user_index=None, last_item_index=None, neg_tar=None, is_training=False):
        feat_dict = None
        user_layer = []
        g.nodes['user'].data['user_h'] = self.user_embedding(g.nodes['user'].data['user_id'].cuda())
        g.nodes['item'].data['item_h'] = self.item_embedding(g.nodes['item'].data['item_id'].cuda())
        if self.layer_num > 0:
            for conv in self.layers:
                feat_dict = conv(g, feat_dict)
                user_layer.append(graph_user(g, user_index, feat_dict['user']))
            item_embed = graph_item(g, last_item_index, feat_dict['item'])
            user_layer.append(item_embed)
        unified_embedding = self.unified_map(torch.cat(user_layer, -1))
        score = torch.matmul(unified_embedding, self.item_embedding.weight.transpose(1, 0))
        if is_training:
            return score
        else:
            neg_embedding = self.item_embedding(neg_tar)
            score_neg = torch.matmul(unified_embedding.unsqueeze(1), neg_embedding.transpose(2, 1)).squeeze(1)
            return score, score_neg

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)



class DGSRLayers(nn.Module):
    def __init__(self, hidden_size, user_max_length, item_max_length, feat_drop=0.2, attn_drop=0.2, K=4):
        super(DGSRLayers, self).__init__()
        self.hidden_size = hidden_size
        self.user_max_length = user_max_length
        self.item_max_length = item_max_length
        self.K = torch.tensor(K).cuda()
        self.agg_gate_u = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.agg_gate_i = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        self.user_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        self.item_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        self.user_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.item_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)

        # attention+ attention mechanism
        self.last_weight_u = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_weight_i = nn.Linear(hidden_size, hidden_size, bias=False)

        self.i_time_encoding = nn.Embedding(user_max_length, hidden_size)
        self.i_time_encoding_k = nn.Embedding(user_max_length, hidden_size)
        self.u_time_encoding = nn.Embedding(item_max_length, hidden_size)
        self.u_time_encoding_k = nn.Embedding(item_max_length, hidden_size)

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
        g = self.graph_update(g)
        g.nodes['user'].data['user_h'] = self.user_update_function(g.nodes['user'].data['user_h'], user_)
        g.nodes['item'].data['item_h'] = self.item_update_function(g.nodes['item'].data['item_h'], item_)
        f_dict = {'user': g.nodes['user'].data['user_h'], 'item': g.nodes['item'].data['item_h']}
        return f_dict

    def graph_update(self, g):
        # user_encoder 对user进行编码
        # update all nodes
        # print(f'graph_update:\ng before: {g}')
        g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func),
                            'pby': (self.item_message_func, self.item_reduce_func)}, 'sum')
        # print(f'g after: {g}\n')
        return g

    def item_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['user_h'] = edges.src['user_h']
        dic['item_h'] = edges.dst['item_h']
        # print(f"item_message_func: \n \
        #         edges: {edges}\n \
        #         time.shape: {dic['time'].shape}\n \
        #         user_h.shape: {dic['user_h'].shape}\n \
        #         item_h.shape: {dic['item_h'].shape}\n")
        return dic

    def item_reduce_func(self, nodes):
        #先根据time排序
        #order = torch.sort(nodes.mailbox['time'], 1)[1]
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] -order -1
        length = nodes.mailbox['item_h'].shape[0]
        #长期兴趣编码
        # if self.item_long == 'orgat':
        e_ij = torch.sum((self.i_time_encoding(re_order) + nodes.mailbox['user_h']) * nodes.mailbox['item_h'], dim=2)\
                /torch.sqrt(torch.tensor(self.hidden_size).float())
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h_long = torch.sum(alpha * (nodes.mailbox['user_h'] + self.i_time_encoding_k(re_order)), dim=1)
        # elif self.item_long == 'gru':
        #     rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
        #     _, hidden_u = self.gru_i(nodes.mailbox['user_h'][torch.arange(length).unsqueeze(1), rnn_order])
        #     h.append(hidden_u.squeeze(0))
        ## 短期兴趣编码
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['user_h'][torch.arange(length), last, :].unsqueeze(1)
        # if self.item_short == 'att':
        e_ij1 = torch.sum(last_em * nodes.mailbox['user_h'], dim=2) / torch.sqrt(
            torch.tensor(self.hidden_size).float())
        alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
        if len(alpha1.shape) == 2:
            alpha1 = alpha1.unsqueeze(2)
        h_short = torch.sum(alpha1 * nodes.mailbox['user_h'], dim=1)
        # elif self.item_short == 'last':
        #     h.append(last_em.squeeze())
        item_h = self.agg_gate_i(torch.cat([h_long, h_short], -1))
        return {'item_h': item_h}

    def user_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['item_h'] = edges.src['item_h']
        dic['user_h'] = edges.dst['user_h']
        # print(f"user_message_func: \n \
        #         edges: {edges}\n \
        #         time.shape: {dic['time'].shape}\n \
        #         user_h.shape: {dic['user_h'].shape}\n \
        #         item_h.shape: {dic['item_h'].shape}\n")
        return dic

    def user_reduce_func(self, nodes):
        # 先根据time排序
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1),1)
        re_order = nodes.mailbox['time'].shape[1] - order -1
        length = nodes.mailbox['user_h'].shape[0]
        # 长期兴趣编码
        # if self.user_long == 'orgat':
        e_ij = torch.sum((self.u_time_encoding(re_order) + nodes.mailbox['item_h']) *nodes.mailbox['user_h'],
                            dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h_long = torch.sum(alpha * (nodes.mailbox['item_h'] + self.u_time_encoding_k(re_order)), dim=1)
        # elif self.user_long == 'gru':
        #     rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
        #     _, hidden_i = self.gru_u(nodes.mailbox['item_h'][torch.arange(length).unsqueeze(1), rnn_order])
        #     h.append(hidden_i.squeeze(0))
        ## 短期兴趣编码
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['item_h'][torch.arange(length), last, :].unsqueeze(1)
        # if self.user_short == 'att':
        e_ij1 = torch.sum(last_em * nodes.mailbox['item_h'], dim=2)/torch.sqrt(torch.tensor(self.hidden_size).float())
        alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
        if len(alpha1.shape) == 2:
            alpha1 = alpha1.unsqueeze(2)
        h_short = torch.sum(alpha1 * nodes.mailbox['item_h'], dim=1)
        # elif self.user_short == 'last':
        #     h.append(last_em.squeeze())
        user_h = self.agg_gate_u(torch.cat([h_long, h_short], -1))
        return {'user_h': user_h}

def graph_user(bg, user_index, user_embedding):
    b_user_size = bg.batch_num_nodes('user')
    # tmp = np.roll(np.cumsum(b_user_size).cpu(), 1)
    # ----numpy写法----
    # tmp = np.roll(np.cumsum(b_user_size.cpu().numpy()), 1)
    # tmp[0] = 0
    # new_user_index = torch.Tensor(tmp).long().cuda() + user_index
    # ----pytorch写法
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1)
    tmp[0] = 0
    new_user_index = tmp + user_index
    # print(f"graph_user:\n \
    #         b_user_size.shape: {b_user_size.shape}\n \
    #         tmp.shape: {tmp.shape}\n \
    #         new_user_index.shape: {new_user_index.shape}\n")
    return user_embedding[new_user_index]

def graph_item(bg, last_index, item_embedding):
    b_item_size = bg.batch_num_nodes('item')
    # ----numpy写法----
    # tmp = np.roll(np.cumsum(b_item_size.cpu().numpy()), 1)
    # tmp[0] = 0
    # new_item_index = torch.Tensor(tmp).long().cuda() + last_index
    # ----pytorch写法
    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    # print(f"graph_item:\n \
    #         b_item_size.shape: {b_item_size.shape}\n \
    #         tmp.shape: {tmp.shape}\n \
    #         new_item_index.shape: {new_item_index.shape}\n")
    return item_embedding[new_item_index]

def order_update(edges):
    dic = {}
    dic['order'] = torch.sort(edges.data['time'])[1]
    dic['re_order'] = len(edges.data['time']) - dic['order']
    return dic