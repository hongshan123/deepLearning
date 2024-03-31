import torch
import torch.nn as nn
from updataASPP import ASPP
import sys  
sys.path.append("deeplabv3+nets")
from deeplabv3_plus import DeepLab
class   Multimodal_unet_plusplus(nn.Module):
    def __init__(self,outchannel):
        super(Multimodal_unet_plusplus,self).__init__()
        self.deeplab=DeepLab(6)
        self.aspp=ASPP(4,outchannel)
        self.inchannel=3
        self.outchannel=outchannel
        self.down1=self.dou_conv(3,64)
        self.down2=self.dou_conv(64,128)
        self.down3=self.dou_conv(128,256)
        self.down4=self.dou_conv(256,512)
        self.down5=self.dou_conv(512,1024)
        self.maxpooling=nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dsm_down1=self.dou_conv(1,64)
        self.dsm_down2=self.dou_conv(64,64)
        self.dsm_down3=self.dou_conv(64,128)
        self.dsm_down4=self.dou_conv(128,256)
        
        self.upx1_2=self.up_conv(64+64,64)
        self.con1=nn.Conv2d(64+3+1,64,stride=1,kernel_size=1)
        self.upx2_2=self.up_conv(128+64,128)
        self.con2=nn.Conv2d(128+64+64,128,stride=1,kernel_size=1)
        self.upx3_2=self.up_conv(256+128,256)
        self.con3=nn.Conv2d(256+128+64,256,stride=1,kernel_size=1)
        self.upx4_2=self.up_conv(512+256,256)
        self.con4=nn.Conv2d(256+256+128,256,stride=1,kernel_size=1)
        
        self.upx1_3=self.up_conv(128,128)
        self.conx1_3=nn.Conv2d(128+64+3+1,128,stride=1,kernel_size=1)
        self.upx2_3=self.up_conv(256,128)
        self.conx2_3=nn.Conv2d(128+128+64+64,128,stride=1,kernel_size=1)
        self.upx3_3=self.up_conv(256,128)
        self.conx3_3=nn.Conv2d(128+256+128+64,128,stride=1,kernel_size=1)
        
        self.upx1_4=self.up_conv(128,64)
        self.convx1_4=nn.Conv2d(64+128+3+1,64,stride=1,kernel_size=1)
        self.upx2_4=self.up_conv(128,64)
        self.convx2_4=nn.Conv2d(64+128+64+64,64,stride=1,kernel_size=1)
        
        self.upx1_5=self.up_conv(64,64)
        self.convx1_5=nn.Conv2d(64+64+3+1,outchannel,stride=1,kernel_size=1)
        self.convout=nn.Conv2d(outchannel*3,outchannel,stride=1,kernel_size=1)
    def forward(self,x,dsm):
        outdeeplab=self.deeplab(x)
        hun=torch.cat((x,dsm),dim=1)
        outaspp=self.aspp(hun)
        x1_1=x#3
        x2_1=self.down1(x1_1)
        x2_1=self.maxpooling(x2_1)#64
        x3_1=self.down2(x2_1)
        x3_1=self.maxpooling(x3_1)#128
        x4_1=self.down3(x3_1)
        x4_1=self.maxpooling(x4_1)#256
        x5_1=self.down4(x4_1)
        x5_1=self.maxpooling(x5_1)#512
        
        dsm1_1=dsm#1
        dsm2_1=self.dsm_down1(dsm1_1)
        dsm2_1=self.maxpooling(dsm2_1)#64
        dsm3_1=self.dsm_down2(dsm2_1)
        dsm3_1=self.maxpooling(dsm3_1)#64
        dsm4_1=self.dsm_down3(dsm3_1)
        dsm4_1=self.maxpooling(dsm4_1)#128
        dsm5_1=self.dsm_down4(dsm4_1)
        dsm5_1=self.maxpooling(dsm5_1)#256
        
        x1_2=self.upx1_2(torch.cat([x2_1,dsm2_1],dim=1))
        x1_2=self.con1(torch.cat([x1_2,x1_1,dsm1_1],dim=1))#64
        x2_2=self.upx2_2(torch.cat([x3_1,dsm3_1],dim=1))
        x2_2=self.con2(torch.cat([x2_2,x2_1,dsm2_1],dim=1))#128
        x3_2=self.upx3_2(torch.cat([x4_1,dsm4_1],dim=1))
        x3_2=self.con3(torch.cat([x3_2,x3_1,dsm3_1],dim=1))#256
        x4_2=self.upx4_2(torch.cat([x5_1,dsm5_1],dim=1))
        x4_2=self.con4(torch.cat([x4_2,x4_1,dsm4_1],dim=1))#256
        
        x1_3=self.upx1_3(x2_2)
        x1_3=self.conx1_3(torch.cat([x1_2,x1_3,dsm1_1,x1_1],dim=1))
        x2_3=self.upx2_3(x3_2)
        x2_3=self.conx2_3(torch.cat([x2_3,x2_2,x2_1,dsm2_1],dim=1))
        x3_3=self.upx3_3(x4_2)
        x3_3=self.conx3_3(torch.cat([x3_3,x3_2,x3_1,dsm3_1],dim=1))
        
        x1_4=self.upx1_4(x2_3)
        x1_4=self.convx1_4(torch.cat([x1_4,x1_3,x1_1,dsm1_1],dim=1))
        x2_4=self.upx2_4(x3_3)
        x2_4=self.convx2_4(torch.cat([x2_4,x2_3,x2_1,dsm2_1],dim=1))
        
        x1_5=self.upx1_5(x2_4)
        x1_5=self.convx1_5(torch.cat([x1_5,x1_4,x1_1,dsm1_1],dim=1))
        out=self.convout(torch.cat((x1_5,outaspp,outdeeplab),dim=1))
        return out
    def dou_conv(self,inchannel,outchannel):
        return nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
    
    def up_conv(self,in_channels,out_channels):
        return nn.Sequential(  
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2,bias=True),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )  
        
    # def up_conv(self,in_channels,out_channels):
    #     return nn.Sequential(
    #         nn.Upsample(scale_factor=2),
    #         nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
    #         nn.BatchNorm2d(out_channels),
    #         nn.ReLU(inplace=True)
    #     )
# mynet=Multimodal_unet_plusplus(6)
# x=torch.randn((1,3,256,256))
# y=torch.randn((1,1,256,256))
# print(mynet(x,y).shape)
