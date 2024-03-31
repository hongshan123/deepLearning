import torch  
import torch.nn as nn  
  
class unet(nn.Module):  
    def __init__(self, n_channels, n_classes, bilinear=False):  
        super(unet, self).__init__()  
        self.n_channels = n_channels  
        self.n_classes = n_classes  
        self.bilinear = bilinear  
          
        self.inc = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)  
        self.relu = nn.ReLU(inplace=True)  
          
        self.down1 = self.double_conv(64, 128)  
        self.down2 = self.double_conv(128, 256)  
        self.down3 = self.double_conv(256, 512)  
          
        self.up1 = self.up_conv(512 + 512, 256)  
        self.up2 = self.up_conv(256 + 256, 128)  
        self.up3 = self.up_conv(128 + 128, 64)  
        self.up4 = self.up_conv(64 + 64, n_classes)  
          
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  
          
        # No need for sigmoid if using cross entropy loss  
        # self.sigmoid = nn.Sigmoid()  
  
    def double_conv(self, in_channels, out_channels):  
        return nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True)  
        )  
      
    def up_conv(self, in_channels, out_channels):  
        if self.bilinear:  
            return nn.Sequential(  
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
                nn.Conv2d(in_channels, out_channels, kernel_size=1)  
            )  
        else:  
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)  
  
    def forward(self, x):  
        x1 = self.relu(self.inc(x))  
        x2 = self.maxpool(x1)  
        x2 = self.relu(self.down1(x2))  
        x3 = self.maxpool(x2)  
        x3 = self.relu(self.down2(x3))  
        x4 = self.maxpool(x3)  
        x4 = self.relu(self.down3(x4))  
          
        x = self.up_conv(x4.size(1) + x3.size(1), 256)(torch.cat([x4, x3], dim=1))  
        x = self.up_conv(x.size(1) + x2.size(1), 128)(torch.cat([x, x2], dim=1))  
        x = self.up_conv(x.size(1) + x1.size(1), 64)(torch.cat([x, x1], dim=1))  
        x = self.up_conv(x.size(1), self.n_classes)(x)  
          
        # Remove sigmoid if using cross entropy loss  
        # x = self.sigmoid(x)  
          
        return x  
  
# Test the network  
x = torch.randn((1, 3, 256, 256))  # Batch size added, was missing before  
print(x.shape)  
mynet = unet(3, 6)  
print(mynet(x).shape)  # Should output the shape of the prediction

