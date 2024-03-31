import tifffile  
import torch
import cv2
img = cv2.imread('Potsdam/train/Label/40203.png')
img_tensor = torch.from_numpy(img).int()
print(img_tensor)
        # trans_label=np.zeros((label.shape[0],label.shape[1]))
        # trans_label[np.all(label==np.array([255.0,255.0,255.0]),axis=-1)]=0
        # trans_label[np.all(label==np.array([0.0,255.0,0.0]),axis=-1)]=1
        # trans_label[np.all(label==np.array([0.0,255.0,255.0]),axis=-1)]=2
        # trans_label[np.all(label==np.array([255.0,0.0,0.0]),axis=-1)]=3
        # trans_label[np.all(label==np.array([255.0,255.0,0.0]),axis=-1)]=4
        # trans_label[np.all(label==np.array([255.0,255.0,255.0]),axis=-1)]=5
