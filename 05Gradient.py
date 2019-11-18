import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d as Axes3D

'''
    a = torch.linspace(-100,100,10)
    print(a)
    
    b = torch.sigmoid(a)
    print(b)
    
    #pytorch autograd
    #y = xw+b                                init b = 0
    x = torch.ones(1)                        tensor([1.])
    print(x)
    w = torch.full([1],2,requires_grad=True) tensor([2.], requires_grad=True)
    print(w)
    mse = F.mse_loss(torch.ones(1),x*w)
    b = torch.autograd.grad(mse,[w])         (tensor([2.]),)
    print(b)
    
    
    #第二种反向传播
    mse.backward()
    w.grad()
    
    # softmax
    a = torch.rand(3)
    a.requires_grad_()
    p = F.softmax(a,dim=0)
    torch.autograd.grad(p[1],[a],retain_graph=True)
    
    #单层感知机
    x = torch.randn(1,10)
    w = torch.randn(1,10,requires_grad=True)
    
    o = torch.sigmoid(x@w.t())
    print(o.shape)               torch.Size([1, 1])
    
    loss = F.mse_loss(torch.ones(1,1),o)
    print(loss.shape)
    loss.backward()
    print(w.grad)
    
    
    #多层感知机

    x = torch.randn(1, 10)
    w = torch.randn(2, 10, requires_grad=True)
    
    o = torch.sigmoid(x @ w.t())
    print(o.shape)                  torch.Size([1, 2])
    
    loss = F.mse_loss(torch.ones(1, 2), o)
    print(loss.shape)                torch.Size([])
    loss.backward()
    print(w.grad)
    
    #链式法则
    x = torch.tensor(1.)
    w1 = torch.tensor(2.,requires_grad=True)
    b1 = torch.tensor(1.)
    w2 = torch.tensor(2.,requires_grad=True)
    b2 = torch.tensor(1.)
    
    
    y1 = x*w1*b1
    y2 = x*w2*b2
    
    
    dy2_dy1 = torch.autograd.grad(y2,[y1],retain_graph=True)[0]
    #反向传播之后这个计算图的内存会被释放，这样就没办法进行第二次反向传播了，所以将retain_graph设置为TRUE
    dy1_dw1 = torch.autograd.grad(y2,[w1],retain_graph=True)[0]
    dy2_dw1 = torch.autograd.grad(y2,[w1],retain_graph=True)[0]
    
    print(dy2_dy1*dy1_dw1)
    print(dy2_dw1)
    两者结果一样
    
    
    ##二维函数优化实例
    def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    x = np.arange(-6,6,0.1)
    y = np.arange(-6,6,0.1)
    print('x,y range:',x.shape,y.shape)        #(120,) (120,)
    X,Y = np.meshgrid(x,y)   #把x的范围和y的范围传入后，生成两张图片
    print('X,Y maps:',X.shape,Y.shape)     #(120, 120) (120, 120)
    Z = himmelblau([X,Y])
    
    
    fig = plt.figure('himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Z)
    ax.view_init(60,-30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    
    
    x = torch.tensor([8.,0.],requires_grad=True)
    optimizer = torch.optim.Adam([x],lr=1e-3)   #设置优化器
    for step in range(20000):                   #开始训练迭代20000次
    
        pred = himmelblau(x)                    #获取预测值
    
        optimizer.zero_grad()                   #清空之前梯度信息
        pred.backward()                         #调用反向传播
        optimizer.step()                        #调用优化器
    
        if step %2000 == 0:                     #每2000次返回打印一次信息
            print('step {}: x = {}, f(x) = {}'.format(step,x.tolist(),pred.item()))  #tolist转化成列表
'''

