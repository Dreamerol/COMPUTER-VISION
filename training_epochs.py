import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# A very important concept in NNs to save the trained model's epochs, afterwards you can average the model_epochs' parameters
# to reduce and minimize the error, minimize the standart deviation(noise)

class ScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(3, 5)
        self.second = nn.ReLU()
        self.third = nn.Linear(5, 1)
        self.four = nn.ReLU()

    def forward(x, self):
        out = self.first(x)
        out = self.second(out)
        out = self.third(out)
        out = self.four(out)
        return out

criterion = nn.MSELoss()
optimizer = optim.SGD()
model = ScorePredictor()

# if we want to save the trained models and their parameters for further operations as avereging model parameters and so on
with torch.no_grad:
    for epoch in (1000):
        prediction = model(X)
        loss = criterion(prediction, Y)
        # clean old gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # save the model
    torch.save(model, f"/home/epochs/{epoch}.pt")
# model = torch.load(PATH) also we can load the model


