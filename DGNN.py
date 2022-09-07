import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.pipeline.sync.skip import skippable, pop, stash
from torch.distributed.pipeline.sync.pipe import PipeSequential

def get_feat(bg, user, batch_idx, ntype, key='h', data=None):
    if not data:
        data = bg.nodes[ntype].data[key]
    if ntype == 'user':
        num_nodes = bg.batch_num_nodes('user')
        tmp = torch.roll(torch.cumsum(num_nodes, 0), 1)
        tmp[0] = 0
        idx = tmp[batch_idx] + user
        return data[idx]
    else:
        return data

def get_batch_mask(bg, user_batch_idx, device=None):
    item_batch_idx = torch.cat([torch.full((n,), i, device=device) 
                                for i, n in enumerate(bg.batch_num_nodes('item'))])
    return (user_batch_idx[:, None] == item_batch_idx[None, :]).to(dtype=torch.float)


class Update(nn.Module):
    
    def __init__(self, hidden_size):
        super(Update, self).__init__()
        self.rnn_weight = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        
    def forward(self, user_new, user_old):
        return F.tanh(self.rnn_weight(torch.cat([user_new, user_old], -1)))


class Reduce(nn.Module):
    
    def __init__(self, etype, max_lookback, hidden_size, attn_drop):
        super(Reduce, self).__init__()
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.atten_drop = nn.Dropout(attn_drop)
        self.norm_const = torch.sqrt(torch.tensor(hidden_size).float())
        self.max_lookback = max_lookback
        self.encode_time = (etype in ['sc', 'cs'])
        if self.encode_time:
            self.key_embed = nn.Embedding(max_lookback, hidden_size)
            self.val_embed = nn.Embedding(max_lookback, hidden_size)

    def forward(self, nodes):
        src = nodes.mailbox['h']
        dst = nodes.mailbox['k']
        query = self.query(dst)
        key = self.key(src)
        val = self.value(src)
        if self.encode_time:
            pred_time = nodes.mailbox['predict_time']
            time = nodes.mailbox['time']
            re_order = pred_time - time - 1
            key_embed = self.key_embed(re_order)
            val_embed = self.val_embed(re_order)
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


class DGNNEmbedding(nn.Module):
    
    def __init__(self, 
                 stack_ntypes : dict, 
                 num_nodes : dict, 
                 hidden_size : int, 
                 device : torch.DeviceObjType,
                 sampling : bool = False):
        super(DGNNEmbedding, self).__init__()
        self.stack_ntypes = stack_ntypes
        self.embeds = nn.ModuleDict({ntype: nn.Embedding(n, hidden_size) for ntype, n in num_nodes.items()})
        self.to(device=device)
        
    def forward(self, g, user, idx):
        for ntype in g.ntypes:
            id = g.nodes[ntype].data[ntype + '_id']
            g.nodes[ntype].data['h'] = self.embeds[ntype](id)
        for ntype in self.stack_ntypes:
            yield stash(ntype, get_feat(g, user, idx, ntype))
        if 'item' not in self.stack_ntypes:
            yield stash('item_embed', self.embeds['item'].weight)
        return g, user, idx
        

class DGNNPredictor(nn.Module):
    
    def __init__(self, 
                 stack_ntypes : list, 
                 num_nodes : int, 
                 hidden_size : int, 
                 layer_num : int, 
                 device : torch.DeviceObjType):
        super(DGNNPredictor, self).__init__()
        self.unified_maps = nn.ModuleDict({ntype: nn.Linear((layer_num + 1) * hidden_size, hidden_size) 
                                           for ntype in stack_ntypes})
        self.stack_ntypes = stack_ntypes
        self.layer_num = layer_num
        self.item_num = num_nodes['item']
        self.device = device
        self.to(device=self.device)
        
    def forward(self, g, user, idx):
        # item_embed /= torch.norm(self.embeds['item'].weight, dim=-1)[..., None]
        g = g.to(self.device)
        idx = idx.to(self.device)
        feat_dict = {}
        for ntype in self.stack_ntypes:
            feat_dict[ntype] = yield pop(ntype)
            for i in range(self.layer_num):
                next_embed = yield pop(ntype + str(i))
                feat_dict[ntype] = torch.cat([feat_dict[ntype], next_embed], -1)
        feat_dict = {ntype: self.unified_maps[ntype](feat) for ntype, feat in feat_dict.items()}
        if 'item' in feat_dict:
            item_feat = feat_dict['item']
            item_idx = g.nodes['item'].data['item_id']
        else:
            item_feat = yield pop('item_embed')
            item_idx = torch.arange(self.item_num, device=self.device)
        score = feat_dict['user'] @ item_feat.transpose(0, 1)
        if 'item' in feat_dict:
            score *= get_batch_mask(g, idx, self.device)
            return score, item_idx
        else:
            return score
    

class DGNNLayer(nn.Module):
    
    def __init__(self, idx, etypes, stack_ntypes, hidden_size, max_lookback, layer_num, devices, feat_drop=0.2, attn_drop=0.2):
        super(DGNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.stack_ntypes = stack_ntypes
        self.idx = idx
        self.device = devices[(idx * len(devices)) // layer_num]
        self.feat_drop = nn.Dropout(feat_drop)
        self.conv, self.reduce, self.update = {}, {}, {}
        for srctype, etype, dsttype in etypes:
            self.conv[srctype] = nn.Linear(hidden_size, hidden_size, bias=False)
            self.reduce[etype] = Reduce(etype, max_lookback, hidden_size, attn_drop)
            self.update[dsttype] = Update(hidden_size)
        self.conv = nn.ModuleDict(self.conv)
        self.reduce = nn.ModuleDict(self.reduce)
        self.update = nn.ModuleDict(self.update)
        self.cross_reduce = CrossReduce()
        self.message = lambda e: {**e.data, 'h': e.src['h'], 'k': e.dst['h']}
        self.to(device=self.device)

    def forward(self, g, user, idx):
        g = g.to(self.device)
        user = user.to(self.device)
        idx = idx.to(self.device)
        feat_dict = {ntype: g.nodes[ntype].data['h'].to(device=self.device) for ntype in g.ntypes}
        for ntype in g.ntypes:
            # print(f'weight: {self.conv[ntype].weight.device}', flush=True)
            # print(f'data  : {feat_dict[ntype].device}', flush=True)
            # print(f'idx   : {self.idx}', flush=True)
            feat_dict[ntype] = self.conv[ntype](self.feat_drop(feat_dict[ntype]))
        update_dict = {etype: (self.message, self.reduce[etype]) for etype in g.etypes}
        g.multi_update_all(update_dict, 'stack', self.cross_reduce)
        for ntype in g.ntypes:
            g.nodes[ntype].data['h'] = self.update[ntype](g.nodes[ntype].data['h'], feat_dict[ntype])
        for ntype in self.stack_ntypes:
            yield stash(ntype + str(self.idx), get_feat(g, user, idx, ntype))
        return g, user, idx


class DGNN(PipeSequential):
    
    def __init__(self, etypes, num_nodes, hidden_size, max_lookback, embed_device, devices, feat_drop=0.2, 
                 attn_drop=0.2, layer_num=3, use_item_feat = True):
        self.layer_num = layer_num
        stack_ntypes = ['user', 'item'] if use_item_feat else ['user']
        
        # prepare arguments
        embed_args = (stack_ntypes, num_nodes, hidden_size)
        args = (etypes, stack_ntypes, hidden_size, max_lookback, layer_num, devices, feat_drop, attn_drop)
        pred_args = embed_args + (layer_num, embed_device)
        embed_args += (embed_device,)
        
        # prepare skip tensor names
        embed_skip = stack_ntypes.copy()
        if 'item' not in stack_ntypes:
            embed_skip.append('item_embed')
        layer_skip = lambda i: [ntype + str(i) for ntype in stack_ntypes]
        all_skip = embed_skip + [name for i in range(0, self.layer_num) for name in layer_skip(i)]
        
        # prepare layer classes
        get_layer_cls = lambda i: skippable(stash=layer_skip(i))(DGNNLayer)
        embed_cls = skippable(stash=embed_skip)(DGNNEmbedding)
        predict_cls = skippable(pop=all_skip)(DGNNPredictor)
        
        # construct layer list
        layers = [embed_cls(*embed_args)]
        layers.extend([get_layer_cls(i)(i, *args) for i in range(0, layer_num)])
        layers.append(predict_cls(*pred_args))
        
        # construct DGNN
        super(DGNN, self).__init__(*tuple(layers))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.xavier_normal_(weight, gain=gain)