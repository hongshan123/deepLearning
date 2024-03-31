import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init  
class   Multimodal_unet(nn.Module):
    def __init__(self,outchannel):
        super(Multimodal_unet,self).__init__()
        self.inchannel=3
        self.outchannel=outchannel
        self.down1=self.dou_conv(3,64)
        self.down2=self.dou_conv(64,128)
        self.down3=self.dou_conv(128,256)
        self.down4=self.dou_conv(256,512)
        self.down5=nn.Conv2d(in_channels=512,out_channels=512,stride=1,kernel_size=1)
        self.up1=self.up_conv(1024+256,256)
        self.up2=self.up_conv(512+128,128)
        self.up3=self.up_conv(256+64,64)
        self.up4=self.up_conv(128+64,64)
        self.out=nn.Conv2d(64,outchannel,kernel_size=1)
        self.maxpooling=nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu=nn.ReLU(inplace=True)
        
        self.dsm_down1=self.dou_conv(1,64)
        self.dsm_down2=self.dou_conv(64,64)
        self.dsm_down3=self.dou_conv(64,128)
        self.dsm_down4=self.dou_conv(128,256)
        self._initialize_weights()  
      
    def _initialize_weights(self):  
        for m in self.modules():  
            if isinstance(m, nn.Conv2d):  
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                if m.bias is not None:  
                    init.constant_(m.bias, 0)  
            elif isinstance(m, nn.ConvTranspose2d):  
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                if m.bias is not None:  
                    init.constant_(m.bias, 0)  
            # 如果还有其他类型的层，可以添加相应的初始化方法    
    def forward(self,x,dsm):
        x1=self.down1(x)
        x1=self.maxpooling(x1)
        dsm1=self.dsm_down1(dsm)
        dsm1=self.maxpooling(dsm1)
        
        x2=self.down2(x1)
        x2=self.maxpooling(x2)
        dsm2=self.dsm_down2(dsm1)
        dsm2=self.maxpooling(dsm2)
        x3=self.down3(x2)
        x3=self.maxpooling(x3)
        dsm3=self.dsm_down3(dsm2)
        dsm3=self.maxpooling(dsm3)
        x4=self.down4(x3)
        x4=self.maxpooling(x4)
        dsm4=self.dsm_down4(dsm3)
        dsm4=self.maxpooling(dsm4)


        y4=self.down5(x4)
        y3=torch.cat([x4,y4,dsm4],dim=1)
        
        y3=self.up1(y3)
        
        y2=torch.cat([y3,x3,dsm3],dim=1)
        y2=self.up2(y2)
        
        y1=self.up3(torch.cat([y2,x2,dsm2],dim=1))
        
        y0=self.up4(torch.cat([y1,x1,dsm1],dim=1))
        y0=self.out(y0)
        return y0
    def dou_conv(self,inchannel,outchannel):
        return nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
    
    # def up_conv(self,in_channels,out_channels):
    #     return nn.Sequential(  
    #         # nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=2,padding=1),
    #         # nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1),
    #         nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True),  
    #         nn.BatchNorm2d(out_channels),  
    #         nn.ReLU(inplace=True)  
    #     ) 
    def up_conv(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )