import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, inCh, outCh, kSize, pad):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=inCh, out_channels=outCh,kernel_size=kSize, padding=pad)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out
    
class ModelExercise(nn.Module):
    def __init__(self):
        super().__init__()
        self.convBl1 = ConvBlock(3, 6, 3, 1)
        self.convBl2 = ConvBlock(6, 4, 3, 1)
        self.avg = nn.AvgPool2d(2)
        self.convBl3 = ConvBlock(4, 2, 3, 1)
        self.conv = nn.Conv2d(2, 4, 3, padding=1)
        self.sigm = nn.Sigmoid()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.finalConv = nn.Sequential(
            ConvBlock(4, 5, 3, 1),
            ConvBlock(5, 2, 3, 1)
        )

    def forward(self, x):
        out = self.convBl1(x)
        out = self.convBl2(out)
        out1 = self.avg(out)
        out1 = self.convBl3(out1)

        out1 = self.conv(out1)
        out1 = self.sigm(out1)
        
        out1 = self.up(out1)
        final = out * out1
        final = self.finalConv(final)
        
        return final


x_data = torch.rand(10, 3, 124, 124)
model = ModelExercise()

y = model(x_data)

