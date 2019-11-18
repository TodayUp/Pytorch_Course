import torch
import numpy as np

'''
    ##基本运算
        add加 sub减  mul乘  div除
    ##矩阵相乘
    * element-wise mul 对应位置元素相乘
    matrix mull：1.torch.mm(2d)  2.torch.matmul(推荐)   3.@  
    
    a = torch.Tensor([[3.,3.],
                      [3.,3.]])
    b = torch.ones(2,2)
    c = torch.matmul(a,b)
    print(c)
    
    ##神经网络线性层实例
    a = torch.rand(4,784)
    x = torch.rand(4,748)
    w = torch.rand(512,784)  #Pytorch权重默认左侧为输出，右侧-为输入
    print((x@w.t()).shape)  
    
    ##2维以上的Tensor
    a = torch.rand(4,3,28,64)
    b = torch.rand(4,3,64,32)
    #c = torch.mm(a,b).shape  #维度报错
    d = torch.matmul(a,b).shape   #matmul只取后边两维运算
    print(d)
    
    
    #次方运算
    pow(a,2/3/4)
    幂次方根 exp/log
    近似值
    .floor()  .ceil()  
    .round()
    .trunc()
    .frac()
    a = torch.tensor(3.14)
    
    print(a.floor())         tensor(3.)     向上取整
    print(a.ceil())          tensor(4.)     向下取整
    print(a.trunc())         tensor(3.)     剪裁整数部分
    print(a.frac())          tensor(0.1400) 剪裁小数部分
    print(a.round())         tensor(3.)      四舍五入
    
    
    #Clamp
    gradient clipping
    w.grad.norm(2)
    grad = torch.rand(3,4)*15

    print(grad.max())        最大值11.3683 
    print(grad.median())     中间值5.1481
    print(grad)
    print(grad.clamp(10))       小于10的置为10 
    print(grad.clamp(0,10))     设置在0到10之间，大于10同理
'''
'''
    ###统计属性
    #范数
    a = torch.full([8],1)
    print(a)                tensor([1., 1., 1., 1., 1., 1., 1., 1.])
    b = a.view(2,4)
    print(b)                tensor([[1., 1., 1., 1.],
                                    [1., 1., 1., 1.]])
    c = a.view(2,2,2)
    print(c)                tensor([[[1., 1.],
                                     [1., 1.]],

                                    [[1., 1.],
                                     [1., 1.]]]) 
    print(a.norm(1))        tensor(8.)
    print(b.norm(1))        tensor(8.)
    print(c.norm(1))        tensor(8.)
    print(a.norm(2))        tensor(2.8284)
    print(b.norm(2))        tensor(2.8284)
    print(c.norm(2))        tensor(2.8284)
    
    print(b.norm(1,dim=1))   tensor([4., 4.])
    print(b.norm(2,dim=1))   tensor([2., 2.])
    print(c.norm(1,dim=0))   tensor([[2., 2.],
                                    [2., 2.]])

    print(c.norm(2,dim=0))   tensor([[1.4142, 1.4142],
                                     [1.4142, 1.4142]])
    #keepdim使维度长度保持一致
    a = torch.rand(4,10)
    print(a.max(dim=1))
    print(a.argmax(dim=1))
    print(a.max(dim=1,keepdim=True))                                                                    

    top-k：返回前K个值
    k-th：返回第k个的值
'''

