import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torchvision import datasets,transforms
from visdom import Visdom
import numpy

#预设参数:训练批次，学习率，和迭代次数
batch_size = 200
learning_rate = 0.01
epochs = 10
viz = Visdom()
#使用Pytorch内置函数，设置训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',train=True,download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,),(0.3081,))
                    ])),   #数据集的保存目录，训练和下载标记，变化换为Tensor，标准化
    batch_size=batch_size,shuffle=True)    #训练批次，打乱数据集

#使用Pytorch内置函数，设置测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',train=False,download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,),(0.3081,))
                    ])),
    batch_size=batch_size,shuffle=True)

#定义一个封装好的类
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,10),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
      x = self.model(x)

      return x
#训练

net = MLP()
optimizer = opt.SGD(net.parameters(),lr=learning_rate)
criteon = nn.CrossEntropyLoss()


for epoch in range(epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        data = data.view(-1,28*28)

        logits = net(data)
        loss = criteon(logits,target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        global_step = 0
        global_step += 1
        viz.line([loss.item()], [global_step], win='train_loss', update='append')
        if batch_idx % 100 ==0:      #每100个返回并打印一次信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),
                      100.*batch_idx/len(train_loader),loss.item()))



    test_loss = 0
    correct = 0
    for data,target in test_loader:
        data = data.view(-1,28*28)

        logits = net(data)
        test_loss += criteon(logits,target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    viz.line([[test_loss, correct / len(test_loader.dataset)]],
             [global_step], win='test', update='append')
    # keshihua
    viz.images(data.view(-1, 1, 28, 28), win='x')
    viz.text(str(pred.detach().cpu().numpy()), win='pred',
             opts=dict(title='pred'))
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss:{:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


print('train:',len(train_db),'test:',len(test_db))
train_db,val_db = torch.utils.data.random_split(train_db,[50000,10000])
print('db1:',len(train_db),'db2:',len(val_db))

train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size,shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_db,
    batch_size=batch_size,shuffle=True)

'''
    x = torch.randn(1,784)
    print(x.shape)
    
    layer1 = nn.Linear(784,200)
    layer2 = nn.Linear(200,200)
    layer3 = nn.Linear(200,10)
    
    x = layer1(x)
    x = F.relu(x)
    print(x.shape)
    x = layer2(x)
    x = F.relu(x)
    print(x.shape)
    x = layer3(x)
    x = F.relu(x)
    print(x.shape)
'''