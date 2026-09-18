import numpy as np
import torch
import torch.nn as nn
from thop import profile 
import sys

'''
TOPs - trillion operation per second - so here we examine three different model architectures and analyze their time for computations
so we can use PIxelShuffle to reduce the spatial dimensions but also increase the channels -> so to reduce the number of computations 
because the kernel will be applied on a smaller area PixelShuffle(scale_factor=2) -> means [B,C,H,W] -> [B, 4C, H/2, W/2]
The 
'''

class ModelOne(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=2)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 2, kernel_size=3, padding=2)
        self.r2 = nn.ReLU()
        self.conv3 = nn.Conv2d(2, 3,kernel_size=3, padding=2)


    def forward(self, x):
        out = self.conv3(self.r2(self.conv2(self.r1(self.conv1(x)))))
        return out

model = ModelOne()
x = torch.randn([1, 3, 64, 64])
flops, params = profile(model, inputs=x)
flops /= 1e12
print(flops)


class ModelTwo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=2)
        self.r1 = nn.ReLU()
        self.p1 = nn.PixelUnshuffle(2)
        self.conv2 = nn.Conv2d(24, 12, kernel_size=3, padding=2)
        self.p2 = nn.PixelShuffle(2)
        self.r2 = nn.ReLU()
        self.conv3 = nn.Conv2d(3, 2,kernel_size=3, padding=2)
        

    def forward(self, x):
        out = self.conv3(self.r2(self.p2(self.conv2(self.p1(self.r1(self.conv1(x)))))))
        return out

model = ModelTwo()
x = torch.randn([1, 3, 64, 64])
flops, params = profile(model, inputs=x)
flops /= 1e12
print(flops)

class ModelThree(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=2)
        self.r1 = nn.ReLU()
        self.p1 = nn.PixelUnshuffle(2)
        self.conv2 = nn.Conv2d(24, 12, kernel_size=3, padding=2)
        self.p2 = nn.PixelShuffle(2)
        self.r2 = nn.ReLU()
        self.p3 = nn.PixelUnshuffle(2)
        self.conv3 = nn.Conv2d(12, 8,kernel_size=3, padding=2)
        self.p4 = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.p4(self.conv3(self.p3(self.r2(self.p2(self.conv2(self.p1(self.r1(self.conv1(x)))))))))
        return out

model = ModelThree()
x = torch.randn([1, 3, 64, 64])
flops, params = profile(model, inputs=x)
flops /= 1e12
print(flops)


class ModelFour(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.r1 = nn.ReLU()
        self.p1 = nn.PixelUnshuffle(2)
        self.conv2 = nn.Conv2d(24, 12, kernel_size=5, padding=2)
        self.p2 = nn.PixelShuffle(2)
        self.r2 = nn.ReLU()
        self.p3 = nn.PixelUnshuffle(2)
        self.conv3 = nn.Conv2d(12, 8,kernel_size=5, padding=2)
        self.p4 = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.p4(self.conv3(self.p3(self.r2(self.p2(self.conv2(self.p1(self.r1(self.conv1(x)))))))))
        return out

model = ModelFour()
x = torch.randn([1, 3, 64, 64])
flops, params = profile(model, inputs=x)
flops /= 1e12
print(flops)


# Model One
# 2.9544048e-05

# Model Two
# 2.5952076e-05

# Model Three TOPs
# 2.2870116e-05

# Model Four
# 5.57056e-05
'''
In conclusion we can say the more the pixel shuffel layers betwen the convolution layers
the faster is the NNs
Thi principle is -> PixelShuffle -> ConvLayer -> PixelUnshuffle
Another interseting observation - if we increase the size of the kernel -> the TOPs rise
the size of the kernel is directly proportional to the TOPs 
'''

# we can write a sript -> we pass the .pt model path and the scripts calculates the TOPs
if __name__ == '__main__':
    model_path = sys.argv[1]
    model = torch.load(model_path, weights_only=False)
    flops, params = profile(model, inputs=x)
    flops /= 1e12
    print(flops)
