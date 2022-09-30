import torch, math
from utils import eval_metric
label = torch.tensor([[ 7., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
                      [ 2.,  0., -1., -1., -1., -1., -1., -1., -1., -1.],
                      [ 8.,  5., -1., -1., -1., -1., -1., -1., -1., -1.],
                      [ 3.,  7., -1., -1., -1., -1., -1., -1., -1., -1.],
                      [ 4.,  9., -1., -1., -1., -1., -1., -1., -1., -1.]])
top = torch.tensor([[6, 8, 9, 5, 2],
                    [8, 0, 2, 4, 9],
                    [1, 2, 3, 7, 5],
                    [9, 8, 7, 1, 5],
                    [1, 8, 6, 3, 0]])
num_pos = torch.tensor([1, 2, 2, 2, 2])
recalls = torch.tensor([[0., 0., 0., 0., 0.],
                        [0., 0.5, 0., 0., 0.],
                        [0., 1., 0.5, 0.5, 0.]]).mean(-1)
dcgs = torch.tensor([[0., 0., 0., 0., 0.],
                      [0., 1 / math.log2(3), 0., 0., 0.],
                      [0., 1 / math.log2(3) + 1 / 2, 1 / math.log2(6), 1 / 2, 0.]])
ns = torch.tensor([1.] + 4 * [1 + 1 / math.log2(3)])[None, :]
ndcgs = (dcgs / ns).mean(-1)
print(f'true recalls: {recalls}')
print(f'true dcgs: {dcgs}')
print(f'true ns: {ns}')
print(f'true ndcgs: {ndcgs}')
results = eval_metric(top, label, num_pos, [1, 2, 5])
print(results)