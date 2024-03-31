import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1=nn.Sequential(
            nn.Conv2d(dim_in,dim_out,kernel_size=1,stride=1,padding=0,dilation=rate,bias=True),
            nn.BatchNorm2d(dim_out,momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2=nn.Sequential(
            nn.Conv2d(dim_in,dim_out,kernel_size=3,stride=1,padding=1,dilation=rate,bias=True),
            nn.BatchNorm2d(dim_out,momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3=nn.Sequential(
            nn.Conv2d(dim_in,dim_out,kernel_size=3,stride=1,dilation=rate*3,padding=rate*3,bias=True),
            nn.BatchNorm2d(dim_out,momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4=nn.Sequential(
            nn.Conv2d(dim_in,dim_out,kernel_size=3,stride=1,dilation=rate*6,padding=rate*6,bias=True),
            nn.BatchNorm2d(dim_out,momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*4, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
    def forward(self, x):
        [b, c, row, col] = x.size()
        conv_x1=self.branch1(x)
        conv_x2=self.branch2(x)
        conv_x3=self.branch3(x)
        conv_x4=self.branch4(x)
        # global_feature = torch.mean(x,2,True)
        # global_feature = torch.mean(global_feature,3,True)
        # global_feature = self.branch5_conv(global_feature)
        # global_feature = self.branch5_bn(global_feature)
        # global_feature = self.branch5_relu(global_feature)
        # global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        feature_cat = torch.cat([conv_x1, conv_x2, conv_x3, conv_x4], dim=1)
        result = self.conv_cat(feature_cat)
        return result
# mynet=ASPP(3,6)
# x=torch.randn((1,3,256,256))
# print(mynet(x).shape)