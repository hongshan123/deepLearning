import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from 模型创建 import LeNet
from 模型创建 import LeNetsequential
from datasetdateloder import rmbdataset
mynet=LeNetsequential(2)
#mynet=LeNet(2)
data_dir="DataLoader与Dataset/data"
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    #transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

rmbdata=rmbdataset(data_dir=data_dir,transform=train_transform)

data_loder=DataLoader(dataset=rmbdata,batch_size=1, shuffle=True)

criterion = nn.CrossEntropyLoss() 

optimizer = optim.SGD(mynet.parameters(), lr=0.001, momentum=0.9)                        # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

numepoch=100
for epoch in range(numepoch):
    loss_mean = 0.
    correct = 0.
    total = 0.
    bad=0
    mynet.train()
    for i,data in enumerate(data_loder):
        
        inputdata,label=data
        #print(imput.shape,label)
        #print(mynet)
        output=mynet(inputdata)
        
        optimizer.zero_grad()
        loss=criterion(output,label)
        loss.backward()
        #更新参数
        optimizer.step()
        #print(output,label)
        #print(output[0][1])
        if(output[0][0]>output[0][1]):
            predict=torch.tensor([0.])
        else:
            predict=torch.tensor([1.])
        if(label==predict):
            correct=correct+1
        else:
            bad+=1
        #print(label,predict)
        total += label.size(0)
        #print("现在是第{}张,loss为:{}".format(i+1,loss))
    print(total)
    print("acc为{},correct为{}".format(correct/total,correct))
    print("Lose为:{}".format(loss))
    #scheduler.step()  # 更新学习率  
        

