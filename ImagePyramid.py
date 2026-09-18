# IMAGE PYRAMIDS - ALGORITHM IN COMPUTER VISION FOR DOWNSAMPLING AN IMAGE WHILE 
# KEEPING THE ESSENTIAL INFORMATION USING KERNELS
import numpy as np
import matplotlib.pyplot as plt
'''
 Image pyramids are used for multiresolutional images -  we get with full/half/quarter.... it is
 like a pyramid size, size/2 .. the idea is to capture fine details, but also see the whole picture
 if we want to locate a big movement
 '''

# applying the Gaussian distribution
# the Gaussian filter is applied to each channel separately
Gaussian_filter = 1/24 * np.array(
                        [[1, 3, 1],
                        [3, 8, 3],
                        [1, 3, 1]])

# Laplacian filter - use dto highlight the intensity of changes
'''
[1 -2  1] + [1
             -2 = it gives us thi intensity of change -> the Laplacian filter  
             1]
I'(x) = I(x + 1) - I(x)
I'(x-1) = I(x) - I(x-1)

I''(x) = I'(x) - I'(x-1) = I(x+1) + I(x-1) -2I(x) -> so we get the 
central pixels by -2 and the neighbouring ones by 1
'''
Laplacian_filter = np.array(
                        [[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])
def applyKernel(image, kernel):
    # here (1,1) ... means add 1 row and 1 column to the first dimension... then  to the secon done
    imagePad = np.pad(image, ((1, 1), (1, 1), (0, 0)), constant_values=0)
    h, w, c = imagePad.shape
    final_image = np.zeros([h, w, c])
    for i in range(0, h-2):
        for j in range(1, w-2):
            final_image[i-1][j-1] = np.sum(imagePad[i: i+3, j: j+3]*kernel, axis=(0, 1))
    return final_image

def imagePyramid(image, kernel):
    h, w, c = image.shape
    if h == 64 and w == 64:
        return image

    imageK = applyKernel(image, kernel)
    new_one = np.zeros([h//2, w//2, c])
    for i in range(1, h , 2):
        for j in range(1, w, 2):
            new_one[(i-1)//2][(j-1)//2] = imageK[i][j]

    # if we want to take the even rows/columns -> on position [2][2] = [4][4]th pixel value
    # for i in range(0, h , 2):
    #    for j in range(0, w, 2):
    #        new_one[i//2][j//2] = imageK[i][j]
    # recursive function for Image Pyramids - we pass the compressed image
    return imagePyramid(new_one, kernel)
    # return new_one
    
image = plt.imread("/home/users/mms00747/Firefox_wallpaper.png")
image = image[:256, :256, :3]

plt.imshow(image)
# print(imagePyramid(image, Gaussian_filter))

import torch
model = torch.load("/home/users/mms00747/train_data/isx3218_2f_0_high/210.pt", weights_only=False)
print(model)
