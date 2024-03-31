import sys  
sys.path.append("deeplabv3+nets")
from deeplabv3_plus import DeepLab
from deeplabv3_training import Dice_loss
import tifffile  
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

image_dir="Potsdam/train/RGB"
label_dir="Potsdam/train/Label"
dsm_dir="Potsdam/train/DSM"
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
    def __init__(self,img_dir,dsm_dir,label_dir,transform=torchvision.transforms.ToTensor):
        
        self.imgs=os.listdir(img_dir)
        self.dsms=os.listdir(dsm_dir)
        self.labels=os.listdir(label_dir)
        self.imgs.sort()
        self.dsms.sort()
        self.labels.sort()
        self.transform=transform
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.dsm_dir=dsm_dir
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        RGBdata=cv2.imread(os.path.join(self.img_dir,self.imgs[index]))
        #print(RGBdata.shape)
        #print(os.path.join(self.img_dir,self.imgs[index]))
        DSMdata=tifffile.imread(os.path.join(self.dsm_dir,self.dsms[index]))
        #print(DSMdata.shape)
        #print(os.path.join(self.dsm_dir,self.dsms[index]))
        label=cv2.imread(os.path.join(self.label_dir,self.labels[index]))
        #print(os.path.join(self.label_dir,self.labels[index]))
        trans_label=np.zeros((label.shape[0],label.shape[1]))
        trans_label[np.all(label==np.array([0.0,0.0,255.0]),axis=-1)]=0
        trans_label[np.all(label==np.array([0.0,255.0,0.0]),axis=-1)]=1
        trans_label[np.all(label==np.array([0.0,255.0,255.0]),axis=-1)]=2
        trans_label[np.all(label==np.array([255.0,0.0,0.0]),axis=-1)]=3
        trans_label[np.all(label==np.array([255.0,255.0,0.0]),axis=-1)]=4
        trans_label[np.all(label==np.array([255.0,255.0,255.0]),axis=-1)]=5
        RGBdata=self.transform(RGBdata).to(device)
        DSMdata=self.transform(DSMdata).to(device)
        label=self.transform(trans_label).to(device)
        #print(label)
        return RGBdata,DSMdata,label

transform=transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        #transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ]
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mydataset=cuntomdataset(image_dir,dsm_dir,label_dir,transform) 
mynet=DeepLab(n_classes)
mynet.to(device)
mynet.train()
optimizer=torch.optim.Adam(mynet.parameters(),lr=1e-4)#优化器
criterion = nn.CrossEntropyLoss()#损失函数
train_loder=DataLoader(mydataset,batch_size=4,shuffle=True)
if __name__=="__main__":
    for epoch in range(10):
       
        i=0
        for x,z,y in train_loder:
            # print(x.shape)
            # x=torch.cat([x,z],dim=1)
            y=y.squeeze(1)
            y=y.long()
            
            
            optimizer.zero_grad()
            y_pred=mynet(x)
    
            loss=criterion(y_pred,y)
            loss.backward()
            optimizer.step()
            i+=1
            if(i%5==0):
                print("现在是第{}轮训练，loss为{}".format(epoch+1,loss))
        # print("现在是第{}轮训练，loss为{}".format(epoch+1,loss))
        torch.save(mynet.state_dict(), '训练参数/deeplabv3+第{}轮.pth'.format(epoch+1))