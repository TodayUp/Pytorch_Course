import torch
import numpy as np
import torchsnooper


'''
随机生成Tensor
#三维tensor，适合于RNN
    a = torch.rand(1,2,3)  #randn服从正太分布，rand服从均匀分布
    print(a)
    print(a.shape)
    print(a[0])
    print(list(a.shape))
#四维张量，适合于CNN
    b = torch.rand(1,3,28,28)
    
    print(b)
    print(b.shape)
    print(b.numel())  数量
    print(b.dim())
    
#创建张量,将numpy数组转换为张量
    c = np.array([2,3.3])
    c = torch.from_numpy(c)
    print(c)
    c = np.ones([2,3])
    c = torch.from_numpy(c)
    print(c)


#导入列表
将列表导入为Tensor
    d = torch.tensor([2.,2.2])
    print(d)
    
    d = torch.FloatTensor([2.,2.2])
'''


'''
生成等差数列等
    a = torch.linspace(0,10,steps=4)
    print(a)
    b = torch.logspace(0,-1,steps=10)
    print(b)
    c = torch.logspace(0,1,steps=10)
    print(c)
'''

'''
随机模块打散
    a = torch.rand(2,3) 
    print(a)
    b = torch.rand(2,2)
    print(b)
    
    idx = torch.randperm(2)  #随机打散
    print(idx)
    print(a[idx])
    print(b[idx])
'''

'''
#Tensor变换
    #扩展维度
    #expend
    a = torch.rand(4,32,14,14)
    #b.shape   torch.Size([1,32,1,1])
                 b.expend(4,32,14,14).shape
                 b.expend(-1,32,-1,-1).shape #-1表示该维度不扩展
    #repeat 
    b.shape    torch.Size([1,32,1,1])
    b.repeat(4,32,1,1).shape   #括号内是拷贝次数
    torch.Size([4,1024,1,1])
    
    b.repeat(4,1,1,1).shape  torch.Size([4,32,1,1])
'''
'''
    ####矩阵转置
        a = torch.randn(3,4)
        print(a)
        print(a.t())  #适用二维
    
    
        #transpose(1,3)包含了需要交换的维度。。。维度的先后顺序 写代码时要注意自己的维度顺序
        a.shape #[4,3,32,32]
        a1 = a.transpose(1,3).view(4,3*32*32).view(4,3,32,32)
        a1 = a.transpose(1,3).contigous().view(4,3*32*32).view(4,3,32,32)
        a2 = a.transpose(1,3).contigous().view(4,3*32*32).view(4,3,32,32).transpose(1,3)
        
        
        #permute,交换维度
        a = torch.rand(4,3,28,28)
        a.transpose(1,3).shape    #torch.Size([4,28,28,3])
        
        b = torch.rand(4,3,28,32)  #torch.Size([4,3,28,32])
        b.transpose(1,3).shape     #torch.Size([4,32,28,3])
        b.transpose(1,3).transpose(1,3).shape  ##torch.Size([4,32,28,3])
        
        b.permute(0,2,3,1).shape   #torch.Size([4,28,32,3])
'''

'''
Broadcast
    ###Tensor Broadcasting: 在运算中，不同大小的两个 array 应该怎样处理的操作。
    # Tensor参数可以自动扩展为相同的大小（不需要复制数据）。
    #如果遵守以下规则，则两个张量是“可播放的”：
    
        #每个张量至少有一个维度。
        #迭代尺寸大小时，从尾随尺寸开始，尺寸大小必须相等，其中一个为1，或者其中一个不存在。
'''
'''
    #Tensor的拼接与拆分 Merge&split
    Cat(连接)  Stack()
    Split()按长度    Chunk()按数量
    
    ###张量的Concate
    两个成绩单
    [class1-4,students,scores]  [4,32,8]
    [class5-9,students,scores]  [5,32,8]
    合并两个成绩单
    a = torch.rand(4,32,8)
    b = torch.rand(5,32,8)
    
    torch.cat([a,b],dim=0).shape   #torch.Size([9,32,8])
    Code:
    a = torch.rand(4, 2)
    print(a)
    b = torch.rand(5, 2)
    print(b)
    c = torch.cat([a, b]（列表）, dim=0)  # torch.Size([9,2])
    print(c)
    
    
    ###张量的stack,维度必须完全一致
    添加新维度，后边维度保持一致
    a = torch.rand(32,8)
    b = torch.rand(32,8)
    
    c = torch.stack([a,b],dim=0)    #torch.Size([2,32,8])
    
    
    ###拆分split、chunk
    split根据长度
    a = torch.rand(32,8)
    b = torch.rand(32,8)
    print(a.shape)
    print(b.shape)
    c = torch.stack([a,b],dim=0)
    print(c.shape)
    aa,bb = c.split([1,1],dim=0)
    print(aa.shape)
    print(bb.shape)
    
    aa,bb = c.split(1,dim=0)
    
    ##chunk按数量
    aa,bb = c.chunk(2,dim=0)
'''
