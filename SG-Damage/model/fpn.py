import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPN(nn.Module):
    def __init__(self, in_channels=1, feature_channels=256,upsampling_size= (600,600)):
        super(FPN, self).__init__()
        self.up_size = upsampling_size

        self.conv1 = ConvBlock(in_channels, 64, kernel_size=3, stride=2)  
        self.conv2 = ConvBlock(64, 128, kernel_size=3, stride=2)  
        self.conv3 = ConvBlock(128, 256, kernel_size=3, stride=2)  
        self.conv4 = ConvBlock(256, 512, kernel_size=3, stride=2)  
        self.latlayer1 = ConvBlock(512, feature_channels, kernel_size=1, padding=0)
        self.latlayer2 = ConvBlock(256, feature_channels, kernel_size=1, padding=0)
        self.latlayer3 = ConvBlock(128, feature_channels, kernel_size=1, padding=0)
        self.latlayer4 = ConvBlock(64, feature_channels, kernel_size=1, padding=0)
        self.toplayer1 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.toplayer2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.toplayer3 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.toplayer4 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.pred1 = nn.Conv2d(feature_channels, 2, kernel_size=1)
        self.pred2 = nn.Conv2d(feature_channels, 2, kernel_size=1)
        self.pred3 = nn.Conv2d(feature_channels, 2, kernel_size=1)
        self.pred4 = nn.Conv2d(feature_channels, 2, kernel_size=1)

    def _upsample_add(self, x, y):
 
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):

        c1 = self.conv1(x)  
        c2 = self.conv2(c1)  
        c3 = self.conv3(c2) 
        c4 = self.conv4(c3)  

        p4 = self.latlayer1(c4)  
        p3 = self._upsample_add(p4, self.latlayer2(c3)) 
        p2 = self._upsample_add(p3, self.latlayer3(c2))  
        p1 = self._upsample_add(p2, self.latlayer4(c1))  


        p4 = self.toplayer1(p4)  
        p3 = self.toplayer2(p3)  
        p2 = self.toplayer3(p2) 
        p1 = self.toplayer4(p1)  


        out_p4 = F.interpolate(self.pred4(p4), size=self.up_size, mode='bilinear', align_corners=True)
        out_p3 = F.interpolate(self.pred3(p3), size= self.up_size, mode='bilinear', align_corners=True)
        out_p2 = F.interpolate(self.pred2(p2), size= self.up_size, mode='bilinear', align_corners=True)
        out_p1 = F.interpolate(self.pred1(p1), size= self.up_size, mode='bilinear', align_corners=True)

        out = (out_p1 + out_p2 + out_p3 + out_p4) / 4

        return out

