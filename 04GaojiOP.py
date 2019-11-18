import torch

'''
    prob = torch.randn(4,10)
    idx = prob.topk(dim=1,k=3)  #topk的输出是当前元素的值以及其对应的索引
    print(idx)

    idx = idx[1]              #0维是值，1维是索引
    print(idx)

    label = torch.arange(10) + 100
    print(label)

    a = torch.gather(label.expand(4,10),dim=1,index=idx.long())
    print(a)
'''
prob = torch.randn(4, 10)
idx = prob.topk(dim=1, k=3)  # topk的输出是当前元素的值以及其对应的索引
print(idx)

idx = idx[0]  # 0维是值，1维是索引
print(idx)