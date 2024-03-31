import torch.nn as nn
import torch.nn.functional as F
import torch
#nn.parameter:张量子类，表示可学习参数，如weight，bias 
#nn.Module() :所有网络层基类，管理网络属性
#nn.function 函数具体实现，如卷积 池化 激活函数等
#nn.init 参数初始化方法
#nn.Module()
class LeNetsequential(nn.Module):
    def __init__(self,classes):
        super(LeNetsequential,self).__init__()
        self.featrues=nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),)
        
        self.classifier=nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,classes),)
    def forward(self,x):
        x=self.featrues(x)
        #x=x.view(x.size()[0],-1)
        x=self.classifier(x)
        return x
class LeNet(nn.Module):
    def __init__(self, classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


class unet(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear=False):
        super(unet,self).__init__()
        self.n_channels=n_channels
        self.n_classes=n_classes
        self.bilinear=bilinear
        
        self.down1=self.double_conv(n_channels,64)
        self.down2=self.double_conv(64,128)
        self.down3=self.double_conv(128,256)
        self.down4=self.double_conv(256,512)
        
        self.up1=self.up_conv(512,256)
        self.up2=self.up_conv(256,128)
        self.up3=self.up_conv(128,64)
        self.up4=self.up_conv(64,n_classes)
        
        
        
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.sigmoid=nn.Sigmoid()

    def double_conv(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
    def up_conv(self,in_channels,out_channels):
        return nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
    
    def forward(self,x):
        x1=self.down1(x)
        print("x1_shape为{}".format(x1.shape))
        x2=self.maxpool(x1)
        print("x2_shape为{}".format(x2.shape))
        x2=self.down2(x2)
        x3=self.maxpool(x2)
        print("x3_shape为{}".format(x3.shape))
        x3=self.down3(x3)
        x4=self.maxpool(x3)
        print("x4_shape为{}".format(x4.shape))
        x4=self.down4(x4)
        
        x=self.up1(x4) 
        print(x.shape) 
        x=torch.cat([x,x3],dim=1)
        print(x.shape)
        x=self.up2(x)
        x=torch.cat([x,x2],dim=1)
        print(x.shape)
        x=self.up3(x)
        x=torch.cat([x,x1],dim=1)
        print(x.shape)
        x=self.up4(x)        
        
        x=self.sigmoid(x)
        
        return x

# x=torch.randn((3,256,256))
# # print
# mynet=unet(3,6)
# print(mynet(x).shape)

class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


"""
    构造下采样模块--右边特征融合基础模块    
"""

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

"""
    模型主架构
"""

class U_Net(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(U_Net, self).__init__()

        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 将e4特征图与d5特征图横向拼接

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)


        return out

# x=torch.randn((1,3,256,256))
# mynet=U_Net(3,6)
# print(mynet(x).shape)

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
        self.up4=self.up_conv(128+64,outchannel)
        self.maxpooling=nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu=nn.ReLU(inplace=True)
        
        self.dsm_down1=self.dou_conv(1,64)
        self.dsm_down2=self.dou_conv(64,64)
        self.dsm_down3=self.dou_conv(64,128)
        self.dsm_down4=self.dou_conv(128,256)
        
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
        y3=self.relu(y3)
        y2=torch.cat([y3,x3,dsm3],dim=1)
        y2=self.up2(y2)
        y2=self.relu(y2)
        y1=self.up3(torch.cat([y2,x2,dsm2],dim=1))
        y1=self.relu(y1)
        y0=self.up4(torch.cat([y1,x1,dsm1],dim=1))
        
        return y0
    def dou_conv(self,inchannel,outchannel):
        return nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
    
    def up_conv(self,in_channels,out_channels):
        return nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        
class   Multimodal_unet_plusplus(nn.Module):
    def __init__(self,outchannel):
        super(Multimodal_unet_plusplus,self).__init__()
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
    def forward(self,x,dsm):
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
        
        return x1_5
    def dou_conv(self,inchannel,outchannel):
        return nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
    
    def up_conv(self,in_channels,out_channels):
        return nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
mynet=Multimodal_unet_plusplus(6)
x=torch.randn((1,3,512,512))
dsm=torch.randn((1,1,512,512))
print(mynet(x,dsm).shape)