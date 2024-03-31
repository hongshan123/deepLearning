from 模型创建 import U_Net
import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
import os
import random
import torchvision.transforms as standard_transforms
import numpy as np
from torch.optim import lr_scheduler
from PIL import Image
from torchvision import transforms

image_dir="posdam/images"
label_dir="posdam/labels"
input_channel=3
n_classes=6

cls_dict={0:[0.0,0.0,255.0],
          1:[0.0,255.0,0.0],
          2:[0.0,255.0,255.0],
          3:[255.0,0.0,0.0],
          4:[255.0,255.0,0.0],
          5:[255.0,255.0,255.0],
}

class cuntomdataset(Dataset):
    def __init__(self,img_dir,label_dir,transform=torchvision.transforms.ToTensor):
        
        self.imgs=os.listdir(img_dir)
        self.labels=os.listdir(label_dir)
        self.imgs.sort()
        self.labels.sort()
        self.transform=transform
        self.img_dir=img_dir
        self.label_dir=label_dir
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        # print(index)
        data=cv2.imread(os.path.join(self.img_dir,self.imgs[index]))
        # print(os.path.join(self.img_dir,self.imgs[index]))
        label=cv2.imread(os.path.join(self.label_dir,self.labels[index]))
        # print(os.path.join(self.label_dir,self.labels[index]))
        trans_label=np.zeros((label.shape[0],label.shape[1]))
        trans_label[np.all(label==np.array([0.0,0.0,255.0]),axis=-1)]=0
        trans_label[np.all(label==np.array([0.0,255.0,0.0]),axis=-1)]=1
        trans_label[np.all(label==np.array([0.0,255.0,255.0]),axis=-1)]=2
        trans_label[np.all(label==np.array([255.0,0.0,0.0]),axis=-1)]=3
        trans_label[np.all(label==np.array([255.0,255.0,0.0]),axis=-1)]=4
        trans_label[np.all(label==np.array([255.0,255.0,255.0]),axis=-1)]=5
        data=self.transform(data)
        return data,transforms.Resize(256)(torch.tensor(trans_label).unsqueeze(0))

transform=transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ]
)

mydataset=cuntomdataset(image_dir,label_dir,transform)

mynet=U_Net(input_channel,n_classes)

optimizer=torch.optim.Adam(mynet.parameters(),lr=0.001,weight_decay=0.005)#优化器
criterion = nn.CrossEntropyLoss()#损失函数
if __name__=="__main__":
    train_loder=DataLoader(mydataset,batch_size=1,shuffle=True)
    for epoch in range(10):
        mynet.train()
        i=0
        for x,y in train_loder:
            # print(x.shape)
            y=y.squeeze(1)
            y=y.long()
            #print(y.shape)
            optimizer.zero_grad()
            y_pred=mynet(x)
    
            loss=criterion(y_pred,y)
            loss.backward()
            optimizer.step()
            i+=1
            if(i%5==0):
                print("loss为:{}".format(loss))
        print("现在是第{}轮训练，loss为{}".format(epoch+1,loss))
        torch.save(mynet, 'unet.pth')