import numpy as np
from numpy import random

from scipy.optimize import minimize
world_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 0],

    [0, 1, 0],
    [1, 1, 0],
    [2, 1, 0],

    [0, 2, 0],
    [1, 2, 0],
    [2, 2, 0]
])

image_points = np.array([
    [120, 80],
    [170, 82],
    [220, 84],

    [118, 130],
    [168, 132],
    [218, 134],

    [116, 180],
    [166, 182],
    [216, 184]
])
print(3)

R_t = np.random.randn(3, 4)
thita = np.random.randn(17)
def callibration(thita):
    
    R_t = thita[0:12].reshape([3, 4])
    
    fx = thita[12]
    fy = thita[13]
    cx = thita[14]
    cy = thita[15]
    s = thita[16]
    base_mat = np.array(
                [[fx, 0, cx],
                [0,fy, cy],
                [0, 0, 1]])


    new_world_points = np.vstack([world_points.T, np.ones((1, world_points.shape[0]))])
    result = (1/s)* base_mat @ R_t @ new_world_points
    result = result.T
    result = np.array([[item[0]/item[2], item[1]/item[2]] for item in result])
    return result

def loss(thita, y):
    result = callibration(thita)
    return np.sum(np.sqrt(np.sum((result-y)**2, axis=1)))

finali = minimize(loss, thita, image_points)
print(finali.x)




