#逻辑回归是线性的二分类模型
#y=f(wx+b) f(x)=1/(1+e^(-x)) 对数几率回归
#wx+b=In(y/(1-y))
#class=0(y<0.5) class=1(y>=0.5)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)#随机数种子

#生成数据
sample_nums=10000
mean_value=1.7
bias=5
n_data=torch.ones(sample_nums,2)
x0=torch.normal(mean_value*n_data,1)+bias#生成服从正态分布的数据
y0=torch.zeros(sample_nums)#生成0作为第一类的标签
x1=torch.normal(-mean_value*n_data,1)+bias
y1=torch.ones(sample_nums)#生成1作为第二类的标签
train_x=torch.cat((x0,x1),0)
train_y=torch.cat((y0,y1),0)
# print(y0)
#构建模型
class LR(nn.Module):
    def __init__(self):
        super(LR,self).__init__()
        self.features=nn.Linear(2,1)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self,x):
        x=self.features(x)
        x=self.sigmoid(x)
        return x

Ir_net=LR()

#选择损失函数

loss_fn=nn.BCELoss()

#选择优化器

Ir=0.01
optimizer=torch.optim.SGD(Ir_net.parameters(),lr=Ir,momentum=0.9)

#模型训练

for i in range(10000):
    #前向传播
    y_pred=Ir_net(train_x)
    # print(y_pred)
    #计算loss
    loss=loss_fn(y_pred.squeeze(),train_y)
    #反向传播
    loss.backward()
    #更新参数
    optimizer.step()
    
    # print("现在是第{}轮训练,loss为{}".format(i+1,loss))
    mask=y_pred.ge(0.5).float().squeeze()
    correct=(mask==train_y).sum()
    acc=correct.item()/train_y.size(0)
    print("现在是第{}轮训练,acc为{}".format(i+1,acc))
    