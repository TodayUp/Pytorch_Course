import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
'''
    rnn = nn.RNN(100,10)
    rnn._parameters.keys()
    print(rnn._parameters.keys())
    
    
    rnn = nn.RNN(input_size=100,hidden_size=20,num_layers=1)
    print(rnn)
    x = torch.randn(10,3,100)  #维度对应：单词个数，batch，单词维度
    out,h = rnn(x,torch.zeros(1,3,20))
    print(out.shape,h.shape)
    #torch.Size([10, 3, 20]) torch.Size([1, 3, 20])
'''
num_time_steps = 50
input_size = 1
hidden_size = 10
output_size = 1
lr = 0.01
epochs =10


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p,mean=0.0,std=0.001)

        self.linear = nn.Linear(hidden_size,output_size)

    def forward(self, x, hidden_prev):
        out,h = self.rnn(x,hidden_prev)

        out = out.view(-1,hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out,hidden_prev

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr)

hidden_prev = torch.zeros(1,1,hidden_size)
for epoch in range(epochs):
    for iter in range(6000):
        start = np.random.randint(3, size=1)[0]
        time_steps = np.linspace(start, start + 10, num_time_steps)
        data = np.sin(time_steps)
        data = data.reshape(num_time_steps, 1)
        x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
        y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

        output,hidden_prev = model(x,hidden_prev)
        hidden_prev = hidden_prev.detach()

        loss = criterion(output,y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 1000 ==0:
            print(" Epoch: {} Iteration: {} loss {}".format(epoch,iter,loss.item()))


    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

predictions = []
input = x[:,0,:]
for _ in range(x.shape[1]):
    input = input.view(1,1,1)
    (pred,hidden_prev) =model(input,hidden_prev)
    input = pred
    predictions.append(pred.detach().numpy().ravel()[0])

x = x.data.numpy().ravel()
y = y.data.numpy()
plt.scatter(time_steps[:-1],x.ravel(),s=90)
plt.plot(time_steps[:-1],x.ravel())

plt.scatter(time_steps[1:],predictions)
plt.show()