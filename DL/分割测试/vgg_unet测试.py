import torch
import sys  
import os 
absolute_path = os.path.abspath('nets')
sys.path.append(absolute_path)  
from unet import Unet
mynet=Unet(6)
state_dict = torch.load('训练参数/单模态vgg_unet第22轮.pth', map_location=torch.device('cpu')) 
mynet.load_state_dict(state_dict)


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
cls_dict2={0:[255.0,0.0,0.0],
          1:[0.0,255.0,0.0],
          2:[255.0,255.0,0.0],
          3:[0.0,0.0,255.0],
          4:[0.0,255.0,255.0],
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
        DSMdata=tifffile.imread(os.path.join(self.dsm_dir,self.dsms[index]))
        #print(DSMdata.shape)
        label=cv2.imread(os.path.join(self.label_dir,self.labels[index]))
        cv2.imshow('Label Image',label)
        print(os.path.join(self.label_dir,self.labels[index]))
        trans_label=np.zeros((label.shape[0],label.shape[1]))
        trans_label[np.all(label==np.array([0.0,0.0,255.0]),axis=-1)]=0
        trans_label[np.all(label==np.array([0.0,255.0,0.0]),axis=-1)]=1
        trans_label[np.all(label==np.array([0.0,255.0,255.0]),axis=-1)]=2
        trans_label[np.all(label==np.array([255.0,0.0,0.0]),axis=-1)]=3
        trans_label[np.all(label==np.array([255.0,255.0,0.0]),axis=-1)]=4
        trans_label[np.all(label==np.array([255.0,255.0,255.0]),axis=-1)]=5
        RGBdata=self.transform(RGBdata).to(device)
        DSMdata=self.transform(DSMdata).to(device)
        label=transforms.Resize(256)(torch.tensor(trans_label).unsqueeze(0)).to(device)
        return RGBdata,DSMdata,label

transform=transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256),
        #transforms.RandomHorizontalFlip(),
        #transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ]
)

mydataset=cuntomdataset(image_dir,dsm_dir,label_dir,transform) 

train_loder=DataLoader(mydataset,batch_size=1,shuffle=False)
if __name__=="__main__":
    for epoch in range(10):
        mynet.train()
        i=0
        for x,z,y in train_loder:
            # print(x.shape)
            y=y.squeeze(1)
            y=y.long()
            #print(y.shape)
            y_pred=mynet(x)
            y_pred_classes = torch.argmax(y_pred, dim=1)
            y_pred_classes_numpy = y_pred_classes.numpy()
            print(y_pred_classes_numpy)
            print(y_pred_classes_numpy.shape)
            img=np.zeros((256,256,3),dtype=np.uint8)
            for i in range(256):
                for j in range(256):
                     for k in range(6):
                         if(y_pred_classes_numpy[0][i][j]==k):
                             img[i][j]=cls_dict2[k]
            pil_image = Image.fromarray(img)   
            pil_image.show()  
            
