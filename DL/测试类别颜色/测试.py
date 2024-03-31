import os
import cv2
import torch
image=cv2.imread("/Users/mm/Desktop/深度学习/posdam/labels/10.png")
cls_dict={0:[0.0,0.0,255.0],
          1:[0.0,255.0,0.0],
          2:[0.0,255.0,255.0],
          3:[255.0,0.0,0.0],
          4:[255.0,255.0,0.0],
          5:[255.0,255.0,255.0],
}
print(image.shape)
list=[0,0,0,0,0,0]
for i in range(600):
    for o in range(600):
        for k in range(6):
            if(image[i][o][0]==cls_dict[k][0] and image[i][o][1]==cls_dict[k][1] and image[i][o][2]==cls_dict[k][2]):
                list[k]+=1
print(list)