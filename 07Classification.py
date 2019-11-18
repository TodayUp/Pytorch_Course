###多分类问题实战
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torchvision import datasets,transforms


#预设参数:训练批次，学习率，和迭代次数
batch_size = 200
learning_rate = 0.01
epochs = 10
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

#初始化权重和偏置
w1 = torch.randn(200,784,requires_grad=True)
b1 = torch.zeros(200,requires_grad=True)
w2 = torch.randn(200,200,requires_grad=True)
b2 = torch.zeros(200,requires_grad=True)
w3 = torch.randn(10,200,requires_grad=True)
b3 = torch.zeros(10,requires_grad=True)
#定义前向训练
def forward(x):
    x = x @ w1.t() + b1
    x = F.relu(x)

    x = x @ w2.t() + b2
    x = F.relu(x)

    x = x @ w3.t() + b3
    x = F.relu(x)

    return x
#使用何凯明的方法初始化权重，对网络的训练是有好处的
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


#定义优化器，使用随机梯度下降算法
#optimizer = torch.optim.Adam
optimizer = torch.optim.SGD([w1,b1,w2,b2,w3,b3], lr=learning_rate)
#选择损失函数计算方式
criteon = torch.nn.CrossEntropyLoss()
#迭代训练
for epoch in range(epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        data = data.view(-1,28*28)   #压平数据，其中-1是自适应的意思，根据数据量和batch调整

        logits = forward(data)       #前向训练得到模型预测值
        loss = criteon(logits,target)#计算loss

        optimizer.zero_grad()        #清空梯度
        loss.backward()              #方向传播

        optimizer.step()             #调用优化器

        if batch_idx % 100 ==0:      #每100个返回并打印一次信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),
                      100.*batch_idx/len(train_loader),loss.item()))

    test_loss = 0                        #初始化测试loss
    correct = 0                          #初始化正确分类的个数
    for data,target in test_loader:      #遍历测试数据集
        data = data.view(-1,28*28)
        logits = forward(data)           #前向训练得到模型预测值
        test_loss += criteon(logits, target).item()        #计算模型Loss值

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()          #计算正确预测的个数

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss:{:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,correct,len(test_loader.dataset),
        100.*correct/len(test_loader.dataset)))