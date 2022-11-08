import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import DeviceObjType
from torch.distributed.pipeline.sync.pipe import PipeSequential, WithDevice
from torch.distributed.pipeline.sync.skip import pop, skippable, stash
from dgl._ffi.base import DGLError

from utils import chunk_list, unchunk_list

EType = tuple[str, str, str]

# class Update(nn.Module):
    
#     def __init__(self, hidden_size):
#         super(Update, self).__init__()
#         self.rnn_weight = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        
#     def forward(self, user_new, user_old):
#         return F.tanh(self.rnn_weight(torch.cat([user_new, user_old], -1)))
    
class SimpleReduce(nn.Module):
    
    def forward(self, nodes):
        src = nodes.mailbox['h']
        dst = nodes.mailbox['k']

class Reduce(nn.Module):
    
    def __init__(self, etype, max_lookback, hidden_size, attn_drop):
        super(Reduce, self).__init__()
        # self.key = nn.Linear(hidden_size, hidden_size)
        # self.query = nn.Linear(hidden_size, hidden_size)
        # self.value = nn.Linear(hidden_size, hidden_size)
        self.atten_drop = nn.Dropout(attn_drop)
        self.norm_const = torch.sqrt(torch.tensor(hidden_size).float())
        self.max_lookback = max_lookback
        self.encode_time = (etype in ['ui', 'iu'])
        if self.encode_time:
            self.key_embed = nn.Embedding(3, hidden_size)
            self.val_embed = nn.Embedding(3, hidden_size)

    def forward(self, nodes):
        src = nodes.mailbox['h']
        dst = nodes.mailbox['k']
        query = dst
        key = src
        val = src
        # query = self.query(dst)
        # key = self.key(src)
        # val = self.value(src)
        if self.encode_time:
            #pred_time = nodes.mailbox['predict_time']
            time = nodes.mailbox['semester']
            #re_order = pred_time - time - 1
            key_embed = self.key_embed(time)
            val_embed = self.val_embed(time)
            key = key + key_embed
            val = val + val_embed
        e_ij = torch.sum(key * query, dim=2) / self.norm_const
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h_long = torch.sum(alpha * val, dim=1)
        return {f'h': h_long}


class CrossReduce(nn.Module):
    
    def forward(self, nodes):
        h = nodes.data['h']
        if h.dim() > 2:
            h = h.sum(-2)
        return {'h': h}
    

class DGNNLayer(nn.Module):
    
    def __init__(self, 
                 etypes : list[EType], 
                 ntypes : list[str], 
                 hidden_size : int, 
                 max_lookback : int, 
                 feat_drop : float, 
                 attn_drop : float):
        super(DGNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.feat_drop = nn.Dropout(feat_drop)
        self.conv, self.reduce = {}, {}
        # self.update = {}
        for _, etype, _ in etypes:
            self.reduce[etype] = Reduce(etype, max_lookback, hidden_size, attn_drop)
        for ntype in ntypes:
            self.conv[ntype] = nn.Linear(hidden_size, hidden_size, bias=False)
            # self.update[ntype] = Update(hidden_size)
        self.conv = nn.ModuleDict(self.conv)
        self.reduce = nn.ModuleDict(self.reduce)
        # self.update = nn.ModuleDict(self.update)
        self.cross_reduce = CrossReduce()
        self.message = lambda e: {**e.data, 'h': e.src['h'], 'k': e.dst['h']}

    def forward(self, g):
        feat_dict = {}
        for ntype in g.ntypes:
            feat_dict[ntype] = g.nodes[ntype].data['h']
            g.nodes[ntype].data['h'] = self.conv[ntype](self.feat_drop(feat_dict[ntype]))
        update_dict = {etype: (self.message, self.reduce[etype]) for etype in g.etypes}
        g.multi_update_all(update_dict, 'stack', self.cross_reduce)
        # for ntype in g.ntypes:
        #    g.nodes[ntype].data['h'] = self.update[ntype](g.nodes[ntype].data['h'], feat_dict[ntype])
        return g


class DGNN(nn.Module):
    
    def __init__(self, 
                 etypes : list[EType], 
                 ntypes : list[int], 
                 num_nodes : dict[str, int], 
                 hidden_size : int, 
                 max_lookback : int, 
                 device : DeviceObjType, 
                 feat_drop : float = 0.2, 
                 attn_drop : float = 0.2, 
                 layer_num : int = 3):
        super(DGNN, self).__init__()
        
        # prepare layers
        args = (etypes, ntypes, hidden_size, max_lookback, feat_drop, attn_drop)
        self.layers = nn.ModuleList([DGNNLayer(*args) for _ in range(layer_num)])
        self.embeds = nn.ModuleDict({ntype: nn.Embedding(n, hidden_size) for ntype, n in num_nodes.items()})
        # self.unified_map = nn.Linear((layer_num + 1) * hidden_size, hidden_size)
        
        # construct DGNN
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.xavier_normal_(weight, gain=gain)
                
    def forward(self, g, user):
        for ntype in g.ntypes:
            g.nodes[ntype].data['h'] = self.embeds[ntype](g.nodes(ntype)) # embeddings for specified node indices for specific node type
        # user_h = g.nodes['user'].data['h'][user, ...]
        for layer in self.layers:
            g = layer(g)
            # layer_h = g.nodes['user'].data['h'][user, ...]
            # user_h = torch.cat([user_h, layer_h], -1)
        layer_h = g.nodes['user'].data['h'][user, ...]
        item_h = self.embeds['item'].weight
        return layer_h @ item_h.transpose(0, 1)