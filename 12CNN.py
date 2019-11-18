import torch
import torch.nn as nn

layer = nn.Conv2d(1,3,kernel_size=3,stride=1,padding=0)
x = torch.rand(1,1,28,28)
out = layer(x)
print(out.shape)  #torch.Size([1, 3, 26, 26])

layer = nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)
x = torch.rand(1,1,28,28)
out = layer(x)
print(out.shape)   #torch.Size([1, 3, 28, 28])

layer = nn.Conv2d(1,3,kernel_size=3,stride=2,padding=1)
x = torch.rand(1,1,28,28)
out = layer(x)
print(out.shape)     #torch.Size([1, 3, 14, 14])

print(layer.weight)
print(layer.weight.shape)
x = out
layer = nn.MaxPool2d(2,stride=2)
layer = nn.AvgPool2d(2,stride=2)

x = torch.rand(100,16,784)
layer = nn.BatchNorm1d(16)
out = layer(x)

print(layer.running_mean)
print(layer.running_var)