import torch
import torch.nn as nn
import torch.nn.functional as F


class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1) // 2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)


class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(SmoothDilatedResidualBlock, self).__init__()
        self.pre_conv1 = ShareSepConv(dilation * 2 - 1)
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.pre_conv2 = ShareSepConv(dilation * 2 - 1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)


    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
        y = self.norm2(self.conv2(self.pre_conv2(y)))
        return F.relu(x + y)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class FA(nn.Module):
    def __init__(self, channel):
        super(FA, self).__init__()
        self.ca = CALayer(channel)
        self.pa = PALayer(channel)

    def forward(self, x):
        out = self.ca(x)
        out = self.pa(out)
        return out+x

class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class GCANet(nn.Module):
    def __init__(self, in_c=4, out_c=3, only_residual=True):
        super(GCANet, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 64, 3, 1, 1, bias=False)  
        self.norm1 = nn.InstanceNorm2d(64, affine=True)  
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)

        self.res1 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res2 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res3 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res4 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res5 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res6 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res7 = ResidualBlock(64, dilation=1)

        self.gate = nn.Conv2d(64 * 3, 3, 3, 1, 1, bias=True)

        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1) 
        self.norm4 = nn.InstanceNorm2d(64, affine=True)

        self.dense_1 = nn.Sequential(
                    ResidualBlock(64),
                    ResidualBlock(64),
                    ResidualBlock(64)
                )

        self.dense_2 = nn.Sequential(
                    ResidualBlock(64),
                    ResidualBlock(64),
                    ResidualBlock(64)
                )

        self.fa = FA(64)
        self.pa = PALayer(64)

        self.deconv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(64, affine=True)

        self.deconv1 = nn.Conv2d(64, out_c, 1)

        self.only_residual = only_residual

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        z1 = y  
        y = F.relu(self.norm2(self.conv2(y)))
        z2 = y
        y1 = F.relu(self.norm3(self.conv3(y)))

        y = self.res1(y1)
        y = self.fa(y)
        y = self.res2(y)
        y = self.fa(y)
        y = self.res3(y)
        y = self.fa(y)
        y2 = self.res4(y)
        y2 = self.fa(y2)
        y = self.res5(y2)
        y = self.fa(y)
        y = self.res6(y)
        y = self.fa(y)
        y3 = self.res7(y)

        gates = self.gate(torch.cat((y1, y2, y3), dim=1)) 
        gated_y = y1 * gates[:, [0], :, :] + y2 * gates[:, [1], :, :] + y3 * gates[:, [2], :, :]
        gated_y = self.pa(gated_y)
        y = F.relu(self.norm4(self.deconv3(gated_y))) 

        y4 = torch.add(y, z2) 
        y = self.dense_1(y4) - y  

        y = F.relu(self.norm5(self.deconv2(y)))  
        y5 = torch.add(y, z1) 
        y = self.dense_2(y5) - y  


        if self.only_residual:
            y = self.deconv1(y) 
        else:
            y = F.relu(self.deconv1(y))

        return y
