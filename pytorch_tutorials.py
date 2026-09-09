import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
#from torchmetrics.classification import MulticlassAccuarcy


# create Tensor

mylist = [1.0, 2.3, 3.8]
ten = torch.tensor(mylist)
#print(ten)

# linear nn - in_features, out_features
net = nn.Linear(in_features=3, out_features=2)
output = net(ten)
# print(net(ten))

# stacking layers
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 7),
    nn.Linear(7, 2)

)

print(model(ten))

# model parametgers

total = 0
for item in model.parameters():
    total += item.numel()

print(total)


# adding activation functions
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 7),
    nn.Linear(7, 2),
    nn.Sigmoid()
    #nn.Softmax(dim=-1)
)
target = torch.tensor([0.0, 1.0])
print(model(ten))

# calculating the loss
prediction = model(ten)
crit = CrossEntropyLoss()

loss = crit(prediction, target)
loss.backward()

print(model[0].weight.grad)
print(model[0].bias.grad)

# optimization 
op = optim.SGD(model.parameters(), lr = 0.001)
op.step()

# animals = pd.read_csv("animals.csv")
# feat = animals.iloc[:, 1, -1] # means get all the rows without the last column

# dataset = TensorDataset(torch.tensor(X), torch.tensor(y)) - converting into tensor dataset
# data = DataLoader(dataset, batch_size=2, shuffle=True)

# Training the model

input_data = torch.tensor([[1.0], [2.0], [3.0]])
output_data = torch.tensor([[3.0], [5.0], [7.0]])

model_one = nn.Sequential(
    nn.Linear(1, 3),
    nn.ReLU(),
    nn.Linear(3, 1),
    nn.ReLU()
)

predicted = model_one(input_data)


criterion = nn.MSELoss()
loss = criterion(predicted, output_data)
loss.backward()
op = optim.Adam(model_one.parameters(), lr=0.01)

training_loss = 0
for epoch in range(100):
    for i in range(len(input_data)):
        op.zero_grad()
        inp = input_data[i]
        tar = output_data[i]
    
        predict = model_one(inp)
        loss = criterion(predict, tar)
        loss.backward()
        op.step()

        # print(loss.item())
        training_loss += loss.item()

print(training_loss/len(input_data)*100)

# Torch metrics
# metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
# for feat, label in dataloader:
#     pred = model(feat)
#     metric.update(outputs, label.argmax(dim=-1))
# metric.compute()

#we can add a dropout means on random it gets randomly neurons with probability higher than 0.5 and pass them forward
# -> for better generalization
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Dropout(p=0.5)
)

# we can add a regulizer
optimizer =optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.02)

# creating our net
class Net(nn.Module):
    def __init__(self):
        # we inherit form the base class
        super().__init__()
        self.fc1 = nn.Linear(5, 6)
        self.fc2 = nn.BatchNorm1d(6)

    def forward(self, x):
        output = nn.functional.relu(self.fc2(self.fc1(x)))
        return output


# Implementing Residual Net with Pytorch H(x) = F(x) - x -> prevents vanishing gradients

class ResNet(nn.Module):
    def __init__(self):
        # we inherit form the base class
        super().__init__()
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.BatchNorm1d(6)
        self.fc3 = nn.Linear(3, 5)
        self.fc4 = nn.ReLU()

    def forward(self, x):
        id = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x + id
        x = self.fc4(x)

        return x
    

def linear(x,w, b):
    return x @ w + b

def ReLU(x):
    if x < 0:
        return 0
    else:
        return x


x =[1,2 ,3]
y = [1, 4, 9]

def ResNetMiniForward():
    w1 = np.random.randn(3, 5)
    b1 = np.random.randn(5, 1)
    out = linear(x, w1,  b1)
    
    w2 = np.random.randn(5, 3)
    b2 = np.random.randn(3, 1)
    out1 = (linear(out, w2, b2))
    out_final = ReLU(out1)
    return (out, out_final, w1,w2,b1,b2)

def ResNetMiniBackward(real_output):
    hidden, predicted, w1,w2,b1,b2 = ResNetMiniForward()
    dy = 2*(predicted - real_output)
    
    db2 = np.sum(dy,axis=1)
    dw2 = hidden.T @ dy

    # db1 = dy @ dw2.T
    db1 = np.sum(dy * dw2.T,axis=1)
    dw1 = dx.T @ dy @ dw2.T 

    dx = dy @ w2.T @ w1.T + dy # G'.(F' + id) -> G'.F' + G'   
    # G'.(F'(x) + x') -> G'.(F' + 1)....

    w1 = np.random.randn(3, 5)
    b1 = np.random.randn(5, 1)
    out = linear(x, w1,  b1)
    
    w2 = np.random.randn(5, 3)
    b2 = np.random.randn(3, 1)
    out1 = (linear(out, w2, b2))

    return ReLU(out1)

