# import torch
# import dgl
# from utils import load_data, StaticData
# import numpy as np
# k = -1
# dses = {key: StaticData(f'static/test_None_50_{k}_False/' + key) for key in ['train', 'test']}
# graph = dgl.load_graphs(f'static/test_None_50_{k}_False/graph')[0][0]
# times, counts = np.unique(graph.edges['by'].data['time'], return_counts=True)
# cum = np.cumsum(np.concatenate([[0], counts]))
# for name, ds in dses.items():
# 	print("=" * 30, name, "=" * 30)
# 	for i in range(len(ds)):
# 		print(np.concatenate([np.array(times)[:, np.newaxis], np.array(cum[:-1])[:, np.newaxis]], 1))
# 		print(f"user: {ds[i][1]['user'].item()}")
# 		print(f"targ: {[x.item() for x in ds[i][1]['target'].squeeze() if x.item() != -1]}")
# 		print(f"last: {[x.item() for x in ds[i][1]['last'].squeeze() if x.item() != -1]}")
# 		print(f"time: {ds[i][1]['time'].item()}")
# 		print(f"edge: {len(ds[i][0][0].edges(etype='by')[0])}")
# 		print(torch.cat([x.unsqueeze(1) for x in ds[i][0][0].edges(etype='pby') + (ds[i][0][0].edges['by'].data['time'],)], dim=1))
# 		print("\n\n\n\n")

# import pandas as pd
# enroll = pd.read_csv('data/Enrollments.csv')
# print(len(enroll))
# enroll['ut'] = list(zip(enroll['user_id'], enroll['time']))
# print(len(enroll['ut'].unique()))
# print(len(enroll) / len(enroll['ut'].unique()))

# import pandas as pd
# import numpy as np
# import dgl
# import torch

# from utils import mkdir_if_not_exist
# data = pd.read_csv('data/Enrollments.csv')
# graph = dgl.load_graphs('static/Enrollments_None_50_-1_False/graph')[0][0]
# mkdir_if_not_exist('static/Enrollments_test')
# train_path = 'static/Enrollments_test/train'
# test_path = 'static/Enrollments_test/test'
# val_path = 'static/Enrollments_test/val'
# t_cutoff = 20
# max_lookback = 50 
# train_num, test_num, val_num = 0, 0, 0
# data = data.rename(columns={'user_id': 'users', 'item_id': 'items'})
# data = data.groupby(['time', 'users'])
# data = pd.DataFrame({'items': data['items'].apply(lambda x: list(x))})
# data['num_items'] = data['items'].apply(lambda x: len(x))
# max_items = data['num_items'].max()

# data['items'] = data.apply(lambda r: [r['items'] + (max_items - r['num_items']) * [-1]], axis=1)
# data['items'] = data['items'].apply(lambda lst: lst[0])
# data = data.reset_index().groupby('time')
# keys = ['users', 'items', 'num_items']
# data = data.apply(lambda r: pd.Series([list(r[k]) for k in keys], index=keys))
# data['num_users'] = data['users'].apply(lambda x: len(x))
# max_users = data['num_users'].max()
# def pad(data, key, padding=-1):
# 	data[key] = data.apply(lambda r: [r[key] + (max_users - r['num_users']) * [padding]], axis=1)
# 	data[key] = data[key].apply(lambda lst: lst[0])
# 	return data
# data = pad(data, 'users')
# data = pad(data, 'items', padding=max_items * [-1])
# data = pad(data, 'num_items')
# for t, row in data.iterrows():
# 	edges = {key: (graph.edges[key].data['time'] < t) & 
# 					(graph.edges[key].data['time'] >= t - max_lookback) 
# 					for key in graph.etypes}
# 	subgraph = dgl.edge_subgraph(graph, edges, relabel_nodes=False)
# 	rel_path = '/' + str(t) + '.bin'
# 	keys = ['items', 'users', 'num_items', 'num_users']
# 	labels = {key: torch.tensor([row[key]]).long() for key in keys}
# 	labels['time'] = torch.tensor([t]).long()
# 	if t == t_cutoff and val_path is not None:
# 		dgl.save_graphs(val_path + rel_path, subgraph, labels)
# 		val_num += 1
# 	elif t <= t_cutoff:
# 		dgl.save_graphs(train_path + rel_path, subgraph, labels)
# 		train_num += 1
# 	else:
# 		dgl.save_graphs(test_path + rel_path, subgraph, labels)
# 		test_num += 1

import dgl
from math import ceil
k = 3
graph = dgl.load_graphs('static/Enrollments_4_50_False_3/graph')[0][0]
users, items = graph.edges('uv', etype='pby')
last_user = users.max()
items += last_user
graph = dgl.graph((users, items))
num_nodes = [[] for khop in range(k + 1)]
num_edges = [[] for khop in range(k + 1)]
for khop in range(k + 1):
	for node in graph.nodes():
		if node.item() <= last_user:
			subgraph = dgl.khop_out_subgraph(graph, node.item(), khop)
			graph_sizes = len(khop.successors(node.item()))
			num_nodes[khop].append(subgraph.num_nodes())
			num_edges[khop].append(subgraph.num_edges())
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, k + 1)
for khop in range(k + 1):
    ax[khop].
# print(dgl.k_hopgraph.num_nodes(""))