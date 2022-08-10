import torch
import torch.nn as nn
import torch.nn.functional as F


class DGNN(nn.Module):
    def __init__(self, etypes, user_num, item_num, input_dim, max_lookback, feat_drop=0.2, 
                 attn_drop=0.2, layer_num=3):
        super(DGNN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = input_dim
        self.layer_num = layer_num

        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size)
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size)
        self.unified_map = nn.Linear(self.layer_num * self.hidden_size, 
                                     self.hidden_size, bias=False)
        self.layers = nn.ModuleList([DGNNLayer(etypes, self.hidden_size, max_lookback, feat_drop, attn_drop)
                                     for _ in range(self.layer_num)])
        self.reset_parameters()

    def forward(self, g, user):
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
        return score

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for name, weight in self.named_parameters():
            # if name != 'unified_map.weight' and len(weight.shape) > 1:
            #     nn.init.xavier_normal_(weight, gain=gain)
            # elif len(weight.shape) > 1:
            #     nn.init.zeros_(weight)
            if weight.dim() > 1:
                nn.init.xavier_normal_(weight, gain=gain)


class Update(nn.Module):
    
    def __init__(self, hidden_size):
        super(Update, self).__init__()
        self.rnn_weight = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        
    def forward(self, user_new, user_old):
        return F.tanh(self.rnn_weight(torch.cat([user_new, user_old], -1)))
    
class Reduce(nn.Module):
    
    def __init__(self, max_lookback, hidden_size, attn_drop):
        super(Reduce, self).__init__()
        self.key_embed = nn.Embedding(max_lookback, hidden_size)
        self.val_embed = nn.Embedding(max_lookback, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.atten_drop = nn.Dropout(attn_drop)
        self.norm_const = torch.sqrt(torch.tensor(hidden_size).float())
        self.max_lookback = max_lookback

    def forward(self, nodes):
        pred_time = nodes.mailbox['predict_time']
        time = nodes.mailbox['time']
        src = nodes.mailbox['h']
        dst = nodes.mailbox['k']
        re_order = pred_time - time - 1
        key_embed = self.key_embed(re_order)
        val_embed = self.val_embed(re_order)
        query = self.query(dst)
        key = self.key(src)
        val = self.value(src)
        e_ij = torch.sum((key_embed + key) * query, dim=2) / self.norm_const
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h_long = torch.sum(alpha * (val_embed + val), dim=1)
        return {f'h': h_long}
    
class CrossReduce(nn.Module):
    
    def forward(self, nodes):
        h = nodes.data['h']
        if h.dim() > 2:
            h = h.sum(-2)
        return {'h': h}

class DGNNLayer(nn.Module):
    def __init__(self, etypes, hidden_size, max_lookback, feat_drop=0.2, attn_drop=0.2):
        super(DGNNLayer, self).__init__()
        self.hidden_size = hidden_size

        self.feat_drop = nn.Dropout(feat_drop)
        self.conv, self.reduce, self.update = {}, {}, {}
        for srctype, etype, dsttype in etypes:
            self.conv[srctype] = nn.Linear(hidden_size, hidden_size, bias=False)
            self.reduce[etype] = Reduce(max_lookback, hidden_size, attn_drop)
            self.update[dsttype] = Update(hidden_size)
        
        self.conv = nn.ModuleDict(self.conv)
        self.reduce = nn.ModuleDict(self.reduce)
        self.update = nn.ModuleDict(self.update)

        self.cross_reduce = CrossReduce()
        self.message = lambda e: {**e.data, 'h': e.src['h'], 'k': e.dst['h']}

    def forward(self, g, feat_dict=None):
        if feat_dict == None:
            feat_dict = {ntype: g.nodes[ntype].data[f'{ntype}_h'] for ntype in g.ntypes}
        else:
            feat_dict = {k: v.cuda() for k, v in feat_dict.items()}
        for ntype in g.ntypes:
            g.nodes[ntype].data['h'] = self.conv[ntype](self.feat_drop(feat_dict[ntype]))
        update_dict = {etype: (self.message, self.reduce[etype]) for etype in g.etypes}
        g.multi_update_all(update_dict, 'stack', self.cross_reduce)
        for ntype in g.ntypes:
            g.nodes[ntype].data['h'] = self.update[ntype](g.nodes[ntype].data['h'], feat_dict[ntype])
        feat_dict = {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}
        return feat_dict
    

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