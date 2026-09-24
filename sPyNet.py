import numpy as np
import torch
import torch.nn as nn

# SPATIAL PYRAMID NET
'''
Inverse warping - to reduce the risk of image holes after applying the motion vectors 
two pixels can go to one place -> so that is why we use inverse warping -> the inverse transformation 
is matrix[x][y] = matrix[dx][dy] but in our case dx,dy = 9.8 may not be whole numbers
so we can get the average value of the nbrs pixels -> 0.8 * up + 0.2 * down_pixel 
'''

def upscale_image(image):
    h, w = image.shape
    new_image = np.zeros([2*h, 2*w])

    for i in range(2*h):
        for j in range(2*w):
            if i % 2 == 0 and j % 2 == 0:
                new_image[i][j] = image[i/2][j/2]
            elif i % 2 == 0:
                ji = np.floor(j/2)
                new_image[i][j] = (image[i/2][ji] + image[i/2][ji + 1])/2
            elif j % 2 == 0:
                ii = np.floor(i/2)
                new_image[i][j] = (image[ii][j/2] + image[ii + 1][j/2])/2

    for i in range(2*h):
        for j in range(2*w):
            if i % 2 == 1 and j % 2 == 0:
                new_image[i][j] = (new_image[i+1][j] + new_image[i-1][j] + new_image[i][j-1] + new_image[i][j+1]) /4
            

    return new_image

'''
Bilinear interpolation -> from [[1 2], [3,4]] -> we get 4x4 and these values are on the corners 
and between them we first find the row/column values and then we find the (B-A)/3 and then we get the row/col values as A + d/3 , A + 2/3d .....
and finally when we have the row/col values we find the values in the inner part of the square and its values by finding the median values
from the row/cols -> [[1,1.33,1.67, 2], [...]....]
If the corner pixels are Q11, Q12, Q21, Q22 then the f(x,y) = Q11*(x-x1)*(y - y1)/[(x2-x1)*(y2-y1)] + Q12.....
the idea is to get the weight of the influence - so we calculate it like the proportion between the rectangular area betwen x,y and x1,y1 or Q11 -> subare/AREA
'''
def bilinear_helper(square):
    new_square = np.zeros([4, 4])
    new_square[0][0] = square[0][0]
    new_square[3][3] = square[1][1]
    new_square[0][3] = square[0][1]
    new_square[3][0] = square[1][0]

    corners = [(0, 3), (3, 0), (0, 0), (3, 3)]
    for i in range(4):
        for j in range(4):
            if (i, j) not in corners: # the smaller the area to the point the more close is the pixel so it will be the most similar to its neighbour - so it will have bigger contribution - so we switch the faces
                new_square[i][j] = (new_square[0][0]*(3 - i)*(3 - j) + new_square[0][3]*(3 - i)*(j - 0)  + new_square[3][0]*(i - 0)*(3-j)+ new_square[3][3]*(i-0)*(j-0))/((3-0)*(3-0))

    return new_square

def upsale_bilinear_optical_map(op_map):
    x = op_map[:, :, 0]
    y = op_map[:, :, 1]

    h, w = x.shape
    new_x = np.zeros([2*h, 2*w])
    new_y = np.zeros([2*h, 2*w])
    
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            new_x[2*i:2*i + 4, 2*j: 2*j+4] = bilinear_helper(x[i:i+2, j:j+2])
            new_y[2*i:2*i + 4, 2*j: 2*j+4] = bilinear_helper(y[i:i+2, j:j+2])

    final = torch.stack([torch.from_numpy(new_x), torch.from_numpy(new_y)], dim=2) # we concatenate the two x,y motion matrixes into one -> to get a HxWx2 result map
    final *= 2 # after interpolating the motions we need to multiply by 2 because if in the scale 4x4 the object has moved 2 pixels 
               # then in the new dimension it will move 2*2 = 4 pixels movement

    return final

def warp(frame, vectors):
    h, w, c = frame.shape
    new_frame = np.zeros([h, w, c])
    for i in range(h):
        for j in range(w):
            mx, my = vectors[i][j]

            mxH  = int(np.floor(mx))
            myH = int(np.floor(my))

            x_offset = mx - mxH
            y_offset = my - myH

            new_frame[i][j] = frame[mxH-1][myH]*(1 - x_offset) + frame[mxH+1][myH]*x_offset +  frame[mxH][myH - 1]*(1 - y_offset) + frame[mxH][myH+1]*y_offset
    return new_frame

# Subnet - we have Conv -> RELU -> ... to get more complex characteristics
class Subnet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=2)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=2)
        self.r2 = nn.ReLU()
        self.conv3 = nn.Conv2d(channels[2], channels[3],kernel_size=3, padding=2)


    def forward(self, x):
        out = self.conv3(self.r2(self.conv2(self.r1(self.conv1(x)))))
        return out

'''
Conception - sPyNet arxiv for optical flow estimation
so we have a SpyNet - consisting of few subNets - its whole idea is the subnet_n-1 passes the result
that is an optical flow map/inverse/ from frame2 -> to frame1 -> the to subNet_n we need to pass an 8-channel tensor
consisiting of -> warped_frame1, real_frame1 and op_map - so the net will start notising patterns and studying
for example if channel1 - channel4 -> this is the R pixel is close to zero and the op_map will be close to zero 
Finally we must get a neural net that predicts the optical flow - motion vectors 
this is very useful computer vision algorithm - if we want to predict the position of a moving object
'''

class sPyNet(nn.Module):
    def __init__(self, subNets : list[Subnet]):
        super().__init__()
        self.models = subNets

    # pyramidFrame1,2 -> these are the image frames in different resolutions -> x2, x4.....
    
    '''
    Whole flow 
    we get the result from the subnet_n-1 -> we upscale the previous optical map -> we warp the frame1 with the upscaled
    optical map -> we concat the real pyramid_frame1 with the warped and the upscaled flow map -> we pass this to the 
    subnet_n -> we get the result flow and then to it we add to the result flow we add the prev_upscaled_flow -> here 
    comes the residual connection
    '''

    def forward(self, frame1, frame2, pyramidFrame1, pyramidFrame2):
        
        for k in range(len(self.models)):
            # Level 0 -> this is the base level with  the least resolution -> so we pass an empty optical map
            if k == 0:
                h, w = frame1.shape
                op = np.zeros([h,w])
                x = torch.stack([frame1, frame2, op], dim=2)
                prev_flow = self.subNet[0].forward(x)         

            else:
                # Here are - Level 1, 2,...
                interpolated_flow = upsale_bilinear_optical_map(prev_flow) # then after going up with one resoluiton we need to upscale the previous flow map 
                warped_image = warp(pyramidFrame2[k], interpolated_flow)
                current = torch.cat(pyramidFrame1[k],warped_image, flow)

                '''
                Here comes the interesting Residual connetcion -> it gets the previous optical flow map -> upscale it 
                to get to the size of the new resulted one and finally it sums them -> so no vanishing gradient problem
                and also the sPyNet keeps information from the previous subnets
                '''

                flow = self.subNet[k].forward(current) # we get the optical map -> which is a matrix of motion vectors per each pixel
                flow = flow + interpolated_flow 
                prev_flow = flow

                # becuase of the inverted warping we want to compare the warped frame2 -> frame1 but warped 
                # after inverse adding of the motion vectors -> so we compare tghe real fram1 from the pyramid with the real one
                

        return flow 
