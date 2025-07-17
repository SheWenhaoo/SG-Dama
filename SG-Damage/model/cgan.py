import torch
from torch import nn
import torch.nn.functional as F



class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, l = x.size()
        y = self.avg_pool(x)  # (b, c, 1), global average pooling along the length dimension
        y = self.fc(y)  # (b, c, 1), channel-wise attention
        return x * y  # Element-wise multiplication for channel-wise reweighting

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride, padding):
        super(MultiScaleConv, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=stride, padding=padding[i],
                          padding_mode='circular'),
                nn.BatchNorm1d(out_channels),  # Add batch normalization
                nn.ReLU(inplace=True)  # Add ReLU activation
            )
            for i, k in enumerate(kernel_sizes)
        ])

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)  # Concatenate along the channel dimension


class Mulit_SE(nn.Module):
    def __init__(self):
        super(Mulit_SE, self).__init__()

        # First multi-scale block
        self.multi_scale1 = MultiScaleConv(
            in_channels=1, out_channels=4, kernel_sizes=[3, 5, 7, 9], stride=1, padding=[1, 2, 3, 4]
        )
        self.se1 = SEBlock(channels=16)
        # Second multi-scale block
        self.multi_scale2 = MultiScaleConv(
            in_channels=16, out_channels=8, kernel_sizes=[4, 6, 8, 10], stride=2, padding=[1, 2, 3, 4]
        )
        self.se2 = SEBlock(channels=32)
        # Third multi-scale block
        self.multi_scale3 = MultiScaleConv(
            in_channels=32, out_channels=16, kernel_sizes=[4, 6, 8, 10], stride=2, padding=[1, 2, 3, 4]
        )
        self.se3 = SEBlock(channels=64)
        # Fourth multi-scale block
        self.multi_scale4 = MultiScaleConv(
            in_channels=64, out_channels=32, kernel_sizes=[4, 6], stride=2, padding=[1, 2]
        )
        self.se4 = SEBlock(channels=64)
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(64 * 25, 200),  # Adjusted input dimension
            nn.ReLU(),
            nn.Linear(200, 200)
        )

    def forward(self, x):
        # x: (batch_size, 1, 200)
        # First block
        x = self.multi_scale1(x)
        x = self.se1(x)
        # Second block
        x = self.multi_scale2(x)
        x = self.se2(x)
        # Third block
        x = self.multi_scale3(x)
        x = self.se3(x)
        # Fourth block
        x = self.multi_scale4(x)
        x = self.se4(x)
        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # Reshape to (batch_size, 200, 1)
        x = x.unsqueeze(2)
        return x

class Generator(nn.Module):
    def __init__(self, input_dim=200, output_channels=1, ngf=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.Conv1dEx =Mulit_SE()
        self.encoder = nn.Sequential(

            nn.ConvTranspose2d(input_dim, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(inplace=True),
            resBlock(ngf * 16),
            resBlock(ngf * 16),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 5, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            resBlock(ngf * 8),
            resBlock(ngf * 8),
        )  # 9*9
        self.middle = nn.Sequential(*[AOTBlock(256, [1, 2, 4, 8]) for _ in range(8)])
        self.scaling1 = nn.Sequential(

            UpConv_2x(ngf * 8, ngf * 8),  # 18*18
            resBlock(ngf * 8),
            resBlock(ngf * 8),
            UpConv_2x(ngf * 8, ngf * 4),  # 36*36
            resBlock(ngf * 4),
            resBlock(ngf * 4),
            UpConv1(ngf * 4, ngf * 4),  # 72*72
        )
        self.scaling2 = nn.Sequential(
            # ---------res block
            resBlock(ngf * 4),
            resBlock(ngf * 4),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            resBlock(ngf * 2),
            resBlock(ngf * 2),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            UpConv2(ngf, ngf // 2),
            nn.Conv2d(ngf // 2, 1, 3, stride=1, padding=1),
   
        )


    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.Conv1dEx(x)
        y = y.view(y.size(0), 200, 1, 1)
        y = self.encoder(y)
        y = self.scaling1(y)
        y = self.middle(y)
        y = self.scaling2(y)
        return y


class UpConv_2x(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UpConv_2x, self).__init__()
        self.flow = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.flow(x)


class resBlock(nn.Module):
    def __init__(self, in_channels):
        super(resBlock, self).__init__()

        self.flow = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.flow(x)
        return x + x1


class UpConv1(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv1, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class UpConv2(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv2, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 4, stride=1, padding=0)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


#----------------------------------------------------------------------------------------------------

class ConditionDecoder(nn.Module):
    def __init__(self, input_dim=200):
        super(ConditionDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, 128 * 37 * 37)  # 首先升维到 37x37
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 37x37 -> 74x74
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 74x74 -> 148x148
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1, padding=0),  # 调整到151x151
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)  # 最后输出到单通道

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 37, 37)  # 将一维向量重塑为 64x37x37 的特征图
        x = self.deconv(x)
        x = self.final_conv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels=1, condition_dim=200):
        super(Discriminator, self).__init__()
        self.condition_decoder = ConditionDecoder(input_dim=condition_dim)

        # 使用谱归一化的卷积层
        self.conv_layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_channels + 1, 64, kernel_size=4, stride=2, padding=1)),
            # 256x256 -> 128x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),  # 128x128 -> 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1))  # 16x16 -> 16x16
        )

    def forward(self, img, condition):

        condition_features = self.condition_decoder(condition) 

        combined_input = torch.cat((img, condition_features), dim=1)  

        combined_input = F.interpolate(combined_input, size=(256, 256), mode='bilinear', align_corners=False)

        output = self.conv_layers(combined_input)

        return output


#--------------------------------------AOT-BLOCK

class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate), nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate), nn.ReLU(True)
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
    def forward(self, x):
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat
