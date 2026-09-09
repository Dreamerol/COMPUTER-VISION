import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2 as cv
import torch.nn as nn
import torch

from scipy.io import loadmat

path = "/home/users/mms00747/Downloads/Train400Depth/depth_sph_corr-10.21op2-p-046t000.mat"

data = loadmat(path)

depth = data["Position3DGrid"]

print("shape:", depth.shape)
print("dtype:", depth.dtype)
print("min:", depth.min())
print("max:", depth.max())

# # FIRST STAGE
# '''
# DepthNet - to predict the depth of every image pixel in a passed image
# U-NET architecture - Encoder + Decoder 
# downsampling + upsampling
# we use skip connections to preserve some features when performing spatial resolution reduction

# '''
# # the model gets B, C, H, W but in our case we want to pass just one image - so B = 1
# class DepthNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # ENCODER
#         self.conv1 = nn.Conv2d(3,16, kernel_size=3, padding=1)
#         self.relu1 = nn.ReLU()
#         self.max1 = nn.MaxPool2d(kernel_size=2)

#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU()
#         self.max2 = nn.MaxPool2d(kernel_size=2)

#         # DECODER
#         self.upi1 = nn.Upsample(scale_factor=2)
#         self.conv3 = nn.Conv2d(64, 16, kernel_size=3, padding=1)

#         self.upi2 = nn.Upsample(scale_factor=2)
#         self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

#     def forward(self, x):
#         out1 = self.conv1(x) # 16 x 256 x 256
#         out = self.relu1(out1)
#         out = self.max1(out)

#         out2 = self.conv2(out) # 32 x 128 x 128
#         out = self.relu2(out2)
#         out = self.max2(out)

#         # skip connections - with torch cat realized - used for keeping essential information the Encoder passes to the Decoder before spatial resolution is restored
#         # torch.cat([x, y], dim=1) means which dimension we are summing the dims dim = 1 [B,C1,H,W] CAT [B,C2,H,W] -> [B, C1 + C2, H, W]
#         out = torch.cat([self.upi1(out), out2], dim=1)
        
#         out = self.conv3(out)

#         out = torch.cat([self.upi2(out), out1], dim=1)
#         out = self.conv4(out)
#         return out
# image = plt.imshow("")
# model = DepthNet()
# t = torch.randn([1, 3, 256, 256])
# out = model(t)[0]
# out = torch.cat(out, image)

# print(out.shape)

# # SECOND STAGE
# def get3dpoints(depth_matrix, K):
#     fx, fy, cx, cy = K
#     x, y = depth_matrix.shape
#     # X, Y, Z, R, G, B
#     points = [((u - cx)*depth_matrix[0][u][v]/fx ,(v - cy)*depth_matrix[0][u][v]/fy ,depth_matrix[0][u][v], depth_matrix[1][u][v], depth_matrix[2][u][v], depth_matrix[3][u][v]) for u in range(x) for v in range(y)]
#     return points


# def transformation(R, T, points):
#     transformed_points = [R @ point[0:4] + T for point in points]
#     rgbs = [point[4:7] for point in points]
#     return transformed_points, rgbs

# def map_pixel_coords(points, K):
#     cx, cy, fx, fy  = K
#     uvs = [(point[0]/point[2]*fx + cx, point[1]/point[2]*fy + cy) for point in points]
#     return uvs


# def drawPic(uvs, rgbs):
#     image = np.zeros([256, 256, 3])
#     for uv in uvs:
#         image[uv[0]][uv[1]] = rgbs[[uv[0]]][uv[1]]
#     plt.imshow(image)
#     return image


