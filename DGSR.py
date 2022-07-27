import torch
import torch.nn as nn
import torch.nn.functional as F


class DGSR(nn.Module):
    def __init__(self, ntypes, etypes, user_num, item_num, input_dim, max_lookback, feat_drop=0.2, 
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
        self.layers = nn.ModuleList([DGSRLayers(ntypes, etypes, self.hidden_size, max_lookback, feat_drop, attn_drop)
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
    def __init__(self, ntypes, etypes, hidden_size, max_lookback, feat_drop=0.2, attn_drop=0.2):
        super(DGSRLayers, self).__init__()
        self.hidden_size = hidden_size
        # self.agg_gate_u = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        # self.agg_gate_i = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        self.weight, self.key_encoder, self.val_encoder, self.rnn_weight = {}, {}, {}, {}
        
        for srctype, etype, dsttype in etypes:
            self.weight[srctype] = nn.Linear(hidden_size, hidden_size, bias=False)
            self.key_encoder[etype] = nn.Embedding(max_lookback, hidden_size)
            self.val_encoder[etype] = nn.Embedding(max_lookback, hidden_size)
            self.rnn_weight[dsttype] = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        
        self.weight = nn.ModuleDict(self.weight)
        self.key_encoder = nn.ModuleDict(self.key_encoder)
        self.val_encoder = nn.ModuleDict(self.val_encoder)
        self.rnn_weight = nn.ModuleDict(self.rnn_weight)
            
        self.conv, self.message, self.reduce, self.cross_reduce, self.update  = {}, {}, {}, {}, {}
        for ntype in ntypes:
            self.update[ntype] = self._get_update_func(self.rnn_weight[ntype])
            self.conv[ntype] = self.weight[ntype]
            self.cross_reduce[ntype] = lambda x: x.sum(-2)
        for srctype, etype, dsttype in etypes:
            self.message[etype] = self._get_message_func(srctype, dsttype)
            self.reduce[etype] = self._get_reduce_func(srctype, dsttype, self.key_encoder[etype], self.val_encoder[etype])

    def _get_message_func(self, srctype, dsttype):
        return lambda edges: {'time': edges.data['time'],
                              f'{srctype}_h': edges.src[f'{srctype}_h'],
                              f'{dsttype}_h': edges.dst[f'{dsttype}_h']}

    def _get_update_func(self, rnn_weight):
        return lambda user_now, user_old: F.tanh(rnn_weight(torch.cat([user_now, user_old], -1)))
    

    def _get_reduce_func(self, srctype, dsttype, val_embed, key_embed):
        norm_const = torch.sqrt(torch.tensor(self.hidden_size).float())
        def reduce_func(nodes):
            order = nodes.mailbox['time']
            src = nodes.mailbox[f'{srctype}_h']
            dst = nodes.mailbox[f'{dsttype}_h']
            re_order = order.max() - order
            
            e_ij = torch.sum((key_embed(re_order) + src) * dst, dim=2) / norm_const
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (val_embed(re_order) + src), dim=1)
            return {f'{dsttype}_h': h_long}
        return reduce_func

    def forward(self, g, feat_dict=None):
        if feat_dict == None:
            feat_dict = {ntype: g.nodes[ntype].data[f'{ntype}_h'] for ntype in g.ntypes}
        else:
            feat_dict = {k: v.cuda() for k, v in feat_dict.items()}
        for ntype in g.ntypes:
            g.nodes[ntype].data[f'{ntype}_h'] = self.conv[ntype](self.feat_drop(feat_dict[ntype]))
        g.multi_update_all({etype: (self.message[etype], self.reduce[etype]) for etype in g.etypes}, 'stack')
        for ntype in g.ntypes:
            new_feat = self.cross_reduce[ntype](g.nodes[ntype].data[f'{ntype}_h'])
            g.nodes[ntype].data[f'{ntype}_h'] = self.update[ntype](new_feat, feat_dict[ntype])
        feat_dict = {ntype: g.nodes[ntype].data[f'{ntype}_h'] for ntype in g.ntypes}
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