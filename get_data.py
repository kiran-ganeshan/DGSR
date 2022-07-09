import torch
import dgl
from utils import load_data, StaticData
import numpy as np
k = -1
dses = {key: StaticData(f'static/test_None_50_{k}_False/' + key) for key in ['train', 'test']}
graph = dgl.load_graphs(f'static/test_None_50_{k}_False/graph')[0][0]
times, counts = np.unique(graph.edges['by'].data['time'], return_counts=True)
cum = np.cumsum(np.concatenate([[0], counts]))
for name, ds in dses.items():
	print("=" * 30, name, "=" * 30)
	for i in range(len(ds)):
		print(np.concatenate([np.array(times)[:, np.newaxis], np.array(cum[:-1])[:, np.newaxis]], 1))
		print(f"user: {ds[i][1]['user'].item()}")
		print(f"targ: {[x.item() for x in ds[i][1]['target'].squeeze() if x.item() != -1]}")
		print(f"last: {[x.item() for x in ds[i][1]['last'].squeeze() if x.item() != -1]}")
		print(f"time: {ds[i][1]['time'].item()}")
		print(f"edge: {len(ds[i][0][0].edges(etype='by')[0])}")
		print(torch.cat([x.unsqueeze(1) for x in ds[i][0][0].edges(etype='pby') + (ds[i][0][0].edges['by'].data['time'],)], dim=1))
		print("\n\n\n\n")

# import pandas as pd
# enroll = pd.read_csv('data/Enrollments.csv')
# print(len(enroll))
# enroll['ut'] = list(zip(enroll['user_id'], enroll['time']))
# print(len(enroll['ut'].unique()))
# print(len(enroll) / len(enroll['ut'].unique()))


# items = pd.DataFrame({'items': df['item_id'].apply(lambda x: list(x))})
# items['num_items'] = items['items'].apply(lambda x: len(x))
# max_items = items['num_items'].max()

# items['items'] = items.apply(lambda r: [r['items'] + (max_items - r['num_items']) * [-1]], axis=1)
# items['items'] = items['items'].apply(lambda lst: lst[0])
# items = items.reset_index().groupby('time')
# keys = ['user_id', 'items', 'num_items']
# items = items.apply(lambda r: pd.Series([list(r[k]) for k in keys], index=keys))


