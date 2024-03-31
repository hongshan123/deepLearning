import torch
import torchvision.transforms as transforms
#transforms.Normalize(mean,std,inplace=False)
#功能：逐channel的对图像进行标准化 output=(input-mean)/std
#mean:各通道均值 std:各通道的标准差 inplace:是否原地操作
#transforms.CenterCrop(size) 功能：中心裁剪
#transforms.RandomCrop(size,padding=None,pad_if_needed=False,fill=0,padding_mode='constant')
#功能:从图片中随机裁剪出尺寸为size的图像
#size 大小   padding：设置填充的大小  为a时，上下左右填充a个像素
#pad_if_need:若图像小于设定的size，则填充

#transforms.RandomResizedCrop(size=224,scale=(0.5,0.5)) size裁剪后的尺寸，scale裁剪的比例

