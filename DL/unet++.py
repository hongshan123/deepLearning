import torch  
import torch.nn as nn  
  
class ConvBlock(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super(ConvBlock, self).__init__()  
        self.conv = nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )  
  
    def forward(self, x):  
        return self.conv(x)  
  
  
class Up(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super(Up, self).__init__()  
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.conv = ConvBlock(in_channels, out_channels)  
  
    def forward(self, x1, x2):  
        x1 = self.up(x1)  
        diffY = x2.size()[2] - x1.size()[2]  
        diffX = x2.size()[3] - x1.size()[3]  
  
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,  
                                    diffY // 2, diffY - diffY // 2])  
        x = torch.cat([x2, x1], dim=1)  
        return self.conv(x)  
  
  
class UNetPlusPlus(nn.Module):  
    def __init__(self, in_channels=3, out_channels=1):  
        super(UNetPlusPlus, self).__init__()  
  
        # Encoder path  
        self.conv_down1 = ConvBlock(in_channels, 64)  
        self.conv_down2 = ConvBlock(64, 128)  
        self.conv_down3 = ConvBlock(128, 256)  
        self.conv_down4 = ConvBlock(256, 512)  
  
        # Bridge  
        self.center = ConvBlock(512, 1024)  
  
        # Decoder path  
        self.up_4_3 = Up(1024 + 512, 512)  
        self.up_3_2 = Up(512 + 256, 256)  
        self.up_2_1 = Up(256 + 128, 128)  
        self.up_1_0 = Up(128 + 64, 64)  
  
        # Output path  
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)  
  
    def forward(self, x):  
        x1 = self.conv_down1(x)  
        x2 = self.conv_down2(self.conv_down2(x1))  
        x3 = self.conv_down3(self.conv_down3(x2))  
        x4 = self.conv_down4(self.conv_down4(x3))  
  
        center = self.center(x4)  
  
        x = self.up_4_3(center, x3)  
        x = self.up_3_2(x, x2)  
        x = self.up_2_1(x, x1)  
        x = self.up_1_0(x, x)  
  
        return self.conv_out(x)  
  
  
# 创建模型实例  
model = UNetPlusPlus(in_channels=3, out_channels=1)  
x=torch.randn(1,3,256,256)
print(model(x).shape)