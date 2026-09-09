import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

image = plt.imread("/home/users/mms00747/Downloads/test.png")

hyperparameters = {
    'patch_size' : 4,
    'image_size' : image.shape[0]

}

def get_patches():
    patches = []
    for i in range(hyperparameters["image_size"]):
        for j in range(hyperparameters["image_size"]):
            vec = image[i:i + hyperparameters["patch_size"], j:j + hyperparameters["patch_size"]].reshape([1, -1])
            patches.append(vec)

    return patches

class PatchEmbedder(nn.Module):
    def __init__(self, image_size, inChannels, patch_size):
        super().__init__()  # out channels -> the number of features in a vector
        out_num = hyperparameters["patch_size"]**2 * 3
        self.first_layer = nn.Conv2d(in_channels=3, out_channels=out_num,
                                      kernel_size=hyperparameters["patch_size"],
                                      stride=hyperparameters["patch_size"])
        self.image_size = image_size
        self.inChannels = inChannels
        self.patch_size = patch_size
        self.first_layer = self.first_layer.reshape(out_num, -1)

    def forward(self, x):
        x = self.first_layer(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class PositionEncoder(nn.Module):
    def __init__(self, vec_size, num_vectors):
        super().__init__(vec_size, num_vectors)
        self.cls_token = nn.Parameter(np.randn(1,1,vec_size))
        self.pos = np.zeros(num_vectors, vec_size)
        for i in range(num_vectors):
            for j in range(vec_size):
                if j % 2 == 0:
                    self.pos[i][j] = np.sin(i/10000**(j))
                else:
                    self.pos[i][j] = np.cos(i/10000**(j-1))

    def forward(self, x):
        tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, tokens), dim=1)
        x = x + self.pos
        return x
    

class AttentionHead(nn.Module):
    def __init__(self, vec_size, head_size):
        super().__init__(vec_size, head_size)
        self.head_size = head_size
        self.query = nn.Linear(vec_size, head_size)
        self.key = nn.Linear(vec_size, head_size)
        self.value = nn.Linear(vec_size, head_size)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = Q @ K.transpose(-2, -1)
        attention /= np.srqt(self.head_size)
        attention = attention @ V
        return attention


class MultiHead(nn.Module):
    def __init__(self, n_heads, vec_size):
        super().__init__(n_heads, vec_size)
        self.n_heads = n_heads
        self.head_size = vec_size // self.n_heads
        self.lin = nn.Linear(vec_size, vec_size)
        self.mah = nn.ModuleList([AttentionHead(vec_size, self.head_size) for _ in range(n_heads)])

    def forward(self, x):
        out = torch.cat([mah(x) for mah in self.mah], dim=-1)
        out = self.lin(out)
        return out 


class Encoder(nn.Module):
    def __init__(self, n_heads, vec_size, alpha=3):
        super().__init__(n_heads, vec_size)
        self.n_heads = n_heads
        self.vec_size = vec_size

        self.ln1 = nn.LayerNorm(vec_size)
        self.mha = MultiHead(self.n_heads, self.vec_size)
        self.ln2 = nn.LayerNorm(vec_size)
        self.help = nn.Sequential([
            nn.Linear(vec_size, vec_size * alpha),
            nn.GELU(),
            nn.Linear(vec_size * alpha, vec_size),
        ])

    def forward(self, x):
        out = x + self.mha(self.ln1(x))
        out = out + self.help(self.ln2(out))
        return out

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, n_heads, vec_size, n_classes):
        super().__init__(image_size, patch_size, n_heads, vec_size)
        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0
        assert vec_size % n_heads == 0
        self.num_vectors = (image_size[0] // patch_size[0])*(image_size[1] // patch_size[1]) + 1

        self.patchCReater = PatchEmbedder(image_size, 3, patch_size)
        self.pos = PositionEncoder(vec_size, self.num_vectors)
        self.layers = nn.Sequential(*[Encoder(n_heads, vec_size) for _ in range(n_heads)])
        self.classifier = nn.Sequential([
            nn.Linear(vec_size, n_classes),
            nn.Softmax(dim=-1)
        ])

    def forward(self, x):
        out = self.patchCReater(x)
        out = self.pos(out)
        out = self.layers(out)
        out = self.classifier(out[:, 0, :])
        return out


# =========================================== #
# ========= ADVANCED ARCHITECTURES ========== #
# =========================================== #

# Custom Layers

hyper_parameters = {
    "number_blocks" : 3,
    "list_channels" : [1, 2, 3],
    "kernel_size" : 3

}
class ConvMini(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(self, in_channels, out_channels, kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels),
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conve(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class WrapperConv(nn.Module):

    def __init__(self, n_blocks, list_ch, ksizeConv, ksizePool):
        super().__init__(n_blocks, list_ch, ksizeConv, ksizePool)
        self.convos = nn.Sequential([ConvMini(list_ch[i], list_ch[i+1], ksizeConv) for i in range(n_blocks)])
        self.pool = nn.MaxPool2d(ksizePool)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        out = self.convos(x)
        out = self.pool(out)
        out = self.sigm(out)
        return out

# model = WrapperConv(hyper_parameters["number_blocks"], hyper_parameters["list_channels"], hyper_parameters["kernel_size"])

class ObjectDoer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __call__(self, x, y):
        return x*y

print(ObjectDoer(3, 4))

# Quantization -> for decreasing the TOPS and the speed of execution of the operations
# quantizing model weights

model = WrapperConv(1, [2, 3], 3)
# we can quantize all them at once
# there is a problem when there are outliers
# lets convert INT8 into 8 bits


# the function creates a copy -> so we can modify the weights in-place when we are in the body of the function
def quantize_all_weights(model):
    maxi = np.max(model.fc.weights)
    s = 512 / maxi
    model.fc.weights = round(model.fc.weights/s)

def quantize_whole_layer(layer):
    maxi = np.max(layer.weights)
    s = 512 / maxi
    layer.weights = round(layer.weights/s)
    return layer

def quantize_whole(model):
    for i in range(len(model.layers)):
        model.layers[i] = quantize_whole_layer(model.layers[i])

def quantize_channel(layer):
    # broadcasting
    maxis = np.max(layer.weights, axis=0)
    ss = 512/maxis
    return round(layer.weights/ ss)

def quantize_per_channel(model):
    model.layers = [quantize_channel(model.layers[i]) for i in range(len(model.layers))]    
    

#we can create more advanced wrapper module for other models

class ModelOne(nn.Module):
    def __init__(self, inCh, outCh, kSize):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=kSize, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out        

class ModelTwo(nn.Module):
    def __init__(self, inCh, outCh, kSize):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=kSize, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.pool((self.conv(x))))
        return out        

class WrapperMore(nn.Module):
    def __init__(self, model_one : nn.Module, model_two : nn.Module):
        super().__init__()
        self.first = nn.AvgPool2d(kernel_size=2)
        self.second = nn.ReLU()
        self.model_one = model_one
        self.model_two = model_two


    def forward(self, x):
        out = self.model_one.forward(x)
        out = self.first(out)
        out = self.second(out)
        out = self.model_two.forward(out)
        return out

input = torch.randn([10, 3, 25, 25])

modelOne = ModelOne(3, 4, 3)
modelTwo = ModelTwo(4, 2, 3)


wrapper = WrapperMore(modelOne, modelTwo)
h = wrapper.forward(input)
print(h.shape)