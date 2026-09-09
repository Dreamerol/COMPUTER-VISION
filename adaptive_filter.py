import numpy as np
from scipy.optimize import minimize

input_image = np.array([
    [[255,   0,   0], [0, 255,   0], [0,   0, 255], [255,255,0],   [255,0,255],   [0,255,255],   [128,128,128], [50,100,150],  [20,40,60],    [10,20,30]],
    [[120, 30, 90],  [80, 200,40],  [10, 60,220],  [255,100,50],   [90,90,90],    [30,150,200],  [70,80,90],     [100,120,140], [200,50,100],  [0,20,40]],
    [[40,80,120],    [60,100,140],  [80,120,160],  [100,140,180],  [120,160,200], [140,180,220], [160,200,240],  [180,220,255], [200,240,255], [220,255,255]],
    [[255,200,100],  [240,180,80],  [220,160,60],  [200,140,40],   [180,120,20],  [160,100,10],  [140,80,5],     [120,60,0],    [100,40,0],    [80,20,0]],
    [[10,10,10],     [20,20,20],    [30,30,30],    [40,40,40],     [50,50,50],    [60,60,60],    [70,70,70],    [80,80,80],   [90,90,90],   [100,100,100]],
    [[15,100,200],   [25,110,210],  [35,120,220],  [45,130,230],   [55,140,240],  [65,150,250],  [75,160,255],   [85,170,255],  [95,180,255],  [105,190,255]],
    [[200,20,30],    [210,30,40],   [220,40,50],   [230,50,60],    [240,60,70],   [250,70,80],   [255,80,90],    [255,90,100],  [255,100,110], [255,110,120]],
    [[5,50,100],     [15,60,110],   [25,70,120],   [35,80,130],    [45,90,140],   [55,100,150],  [65,110,160],   [75,120,170],  [85,130,180],  [95,140,190]],
    [[255,255,255],  [240,240,240],  [225,225,225],  [210,210,210],  [195,195,195], [180,180,180], [165,165,165], [150,150,150], [135,135,135], [120,120,120]],
    [[0,0,0],        [10,0,20],     [20,10,30],     [30,20,40],     [40,30,50],   [50,40,60],   [60,50,70],    [70,60,80],   [80,70,90],   [90,80,100]]
])

output_image = np.random.randn(8, 8)

size = input_image.shape[0]

weights = np.random.randn(3*3*3)
Final_weights = []

def loss(weights, input_matrix, output):
    
    weights_new = weights.reshape(3,3,3)
    # for i in range(size-2):
    #     for j in range(size-2):
    result = np.sum(weights_new * input_matrix)
    return (result - output)**2

for i in range(0, size-2):
    for j in range(0, size-2):
        
        result = minimize(loss, weights, args=(input_image[i:i+3, j:j+3, :], output_image[i][j]))
        Final_weights.append(result.x)

# print(Final_weights)

# matrix[i:i+3][j:j+3] -> means first we get m1 = matrix[i:i+3] and then m2 = m1[j:j+3] again it slices from the same dimension

# second way - based on the gradient we choose whether to use sharpening filter or box filter for reducing the noise

box_filter = 1/9*np.ones([3, 3])
sharp_filter = np.array([[0, -1, 0], 
                         [-1, 5,-1], 
                         [0, -1, 0]])

gradientY = np.array([[-1, 0, 1],
                       [-2, 0, 2], 
                       [-1, 0, 1]])

gradientX = np.array([[-1, -2, -1], 
                      [0, 0, 0], 
                      [1, 2, 1]])

def fromRGBtoGrayscale(channels):
    R = channels[0]
    G = channels[1]
    B = channels[2]
    return 0.299*R + 0.587*G + 0.114*B


def grayscaleConverter(input_image):
    h, w, c = input_image.shape
    out = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            out[i][j] = fromRGBtoGrayscale(input_image[i][j])
    return out

def findGrad(image):
    Gx = np.sum(gradientX * image)
    Gy = np.sum(gradientY * image)

    G = np.sqrt(Gx**2 + Gy**2)
    return G

def findGradients(image):
    gr_im = grayscaleConverter(image)
    h, w = gr_im.shape
    output = np.zeros([h-3, w-3])
    for i in range(h-3):
        for j in range(w-3):
            current = gr_im[i:i+3, j:j+3]
            G = findGrad(current)
            filter = box_filter if G < 20 else sharp_filter
            output[i][j] = np.sum(filter * current)
    return output


print(findGradients(input_image))


