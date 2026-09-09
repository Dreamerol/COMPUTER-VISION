import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2 as cv

Gy = [[-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]]

Gx = [[-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]]

# calculating image gradients
def compute_I(old_image):
    x,y = old_image.shape
    Ix = np.zeros([x, y])
    Iy = np.zeros([x, y])

    for i in range(1, x-1):
        for j in range(1, y-1):
            Ix[i][j] = np.sum(old_image[i-1:i+2, j-1:j+2]*Gx)
            Iy[i][j] = np.sum(old_image[i-1:i+2, j-1:j+2]*Gy)
    return Ix, Iy

def loss(A, It, vec):
    dx, dy = vec
    loss = np.mean((A[:, 0]* dx + A[:, 1] * dy + It)**2) # -> [(Ix11.dx + Iy11.dy + It11)**2]
    return loss


def compute_gradient(old_image, new_image, x, y):
    It = new_image - old_image
    Ix, Iy = compute_I(old_image)

    Ix_window = Ix[x-1:x+2, y-1:y+2]
    Iy_window = Iy[x-1:x+2, y-1:y+2]
    It_window = It[x-1:x+2, y-1:y+2]

    '''
    we get the image gradients - Ix and Iy 
    the formula is Ix.dx + Iy.dy + It = 0
    we use matrixes to represent this equation and we want to find the 

    [[Ix11 Iy11],         [It11,   
    [Ix12 Iy12], * [dx, = It12
                    dy]      ...]
                    
    ....]
    we want to solve the optimization task to find those dx, dy, so the eij vectors are the least e11 := (Ix11.dx + Iy11.dy -It11)**2 ..  and we find the sum we want to be the smallest
    
    '''
    v = np.zeros([2, 1])
    A = np.column_stack([Ix_window.flatten(), Iy_window.flatten()])
    I = It_window.flatten()
    minimize(loss, v, args=(A, I))

    return v

cap = cv.VideoCapture("https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4")

flag, frame = cap.read()
cv.imshow("frame", frame)
old_frame = frame

while True:
    flag, frame = cap.read()
    if not flag:
        break
    cv.imshow("frame", frame)
    cv.colorbgr
    dx, dy = compute_gradient(old_frame, frame, x, y)
    old_frame = frame


    

