import os, torch, dgl, pickle, pandas as pd, numpy as np
from torch.distributed import rpc
from torch.distributed.pipeline.sync.pipe import Pipe
from collections import OrderedDict
from DGNN import DGNN
from IPython.display import display
from utils import subsample, get_topk_items, multihot

fake_user = False

# data
max_lookback = 12
test_num = 4
data_id = f"Enrollments_{test_num}_{max_lookback}_False"
run_id = ''
name = None
old_run = False

# model
margin = False
sampling = False
feat = False
multiplier = 1
hypers = OrderedDict([
	('bs', 2),
	('lr', 0.001),
	('ep', 30),
	('l2', 0.0001),
	('pw', 1.0),
	('ft', 0.0),
	('at', 0.2),
	('ln', 3),
	('hs', 50),
	# ('mu', 25),
	# ('mi', 25),
	# ('k', 3)
])
# if not sampling:
#     hypers = hypers[:-3]

# eval
ats = [5, 10, 20]
neg_num = 100

# retrieve model state
if old_run:
    data_id = 'old_runs'
    run_id = name
else:
    suffix = '_'.join([f'{key}{val}' for key, val in hypers.items()])
    suffix = suffix + '_' + ('margin' if margin else 'bce')
    if not run_id:
        run_id = suffix
    else:
        run_id = run_id + '_' + suffix
    run_id = run_id + '_' + ('feat' if feat else 'embed')
print(run_id)
state = torch.load(f"static/{data_id}/{run_id}/model")
with open(f"static/{data_id}/meta", 'rb') as f:
    meta = pickle.load(f)
    user_num = meta['user_num']
    item_num = meta['item_num']
    etypes = meta['etypes']
    
# load model and graphs
devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]     # get available devices
if len(devices) > hypers['ln'] + 2:                                                 # use at most layer_num + 2 devices
    devices = devices[:hypers['ln'] + 2]
device = devices[-1]                                                                # device to embed and predict on
devices = devices[:-1]                                                              # devices to compute graph layers on
torch.cuda.set_device(device)                                                       # redirect .cuda() to correct device
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29600'
rpc.init_rpc('worker', rank=0, world_size=1)
args = (etypes, {'user': user_num, 'item': item_num}, hypers['hs'], max_lookback, device, devices, hypers['ft'], hypers['at'], hypers['ln'], feat)
model = Pipe(DGNN(*args), 1, 'never')
model.load_state_dict(state)
graphs, labels = dgl.load_graphs(f"static/{data_id}/test/23.bin")
graph = graphs[0]

# collate
if fake_user:
    graph.add_nodes(1, ntype='user')
    user = torch.tensor([graph.num_nodes('user') - 1])
    predict_time = torch.tensor([23])
    past_items = torch.tensor([])
    past_times = torch.tensor([])
    edges = (user.repeat(len(past_items)), past_items)
    e_data = {'time': past_times, 'predict_time': predict_time.repeat(len(past_items))}
    graph.add_edges(edges, data=e_data, ntype='item')	
else:
    idx = torch.randint(labels['users'].shape[0], (1,))
    user = labels['users'][idx]
    label = labels['items'][idx, :]
    num_target = labels['num_items'][idx]
target = multihot(label, num_target, item_num)
if sampling:
    samples = [subsample(graph, user, hypers['mi'], hypers['mu'], hypers['k']) for _ in range(multiplier)]
    graph, u = zip(*samples)
    user = torch.cat(u).long()
else:
    graph = [graph]
batch = dgl.batch(graph)

# run model and get topk items

user = user.cuda()
idx = torch.zeros((1,)).long().cuda()
device = torch.get_device(user)
# model = model.to(device)
batch = batch.to(device)
target = target.cuda()
num_taret = num_target.cuda()
all_sample_top = {}
with torch.no_grad():
	if feat:
		score, item_idx = model(batch, user, idx).local_value()
	else:
		score = model(batch, user, idx).local_value()
		item_idx = torch.arange(item_num, device=device)
	#score = score.reshape(-1, multiplier, item_num).mean(1)
	top, sample_top = get_topk_items(score, item_idx, target, num_target, max(ats), neg_num)
	for n in range(100, 1000, 100):
		_, s = get_topk_items(score, item_idx, target, num_target, max(ats), n)
		all_sample_top[n] = s.squeeze().cpu()
	top = top.squeeze().cpu()
	sample_top = sample_top.squeeze().cpu()

# get item descriptions
cid_desc = pd.read_csv('data/ucb_raw_data/courses.csv')
abbr_desc = pd.read_csv('data/ucb_raw_data/course_catalog_description.tsv', sep='\t')
courses = cid_desc.merge(abbr_desc, left_on='course desc', right_on='course_description', how='left')
courses = courses[['cid', 'abbr_cid', 'course_title', 'course desc']].sort_values('cid')
def reorder(courses, top):
    courses = courses.copy()
    top = top.cpu().numpy()
    courses = courses[courses['cid'].isin(top.tolist())]
    matrix = (courses['cid'].values[:, None] == top[None, :])
    ranks = np.arange(0, len(top))[None, :]
    courses['idx'] = np.argmin((1. - matrix) * len(top) + ranks, -1)
    return courses.sort_values('idx').drop(columns='idx')
if not fake_user:
    print("positive samples")
    display(courses[courses['cid'].isin(label.tolist()[0])])
print("\n\nall samples")
display(reorder(courses, top))
# print("100 samples")
# display(courses[courses['cid'].isin(sample_top.tolist())])
for key, t in all_sample_top.items():
    print(f"\n\n{key} samples")
    display(reorder(courses, t))
#print(plt.hist(torch.norm(model.item_embedding.weight, dim=-1).tolist()))