import torch
import torch.nn as nn
import torch.nn.functional as F          # adds some efficiency

#Create the Multilayer Perceptron
#Input Size = 1024 (32*32px Images), Output Size = 10 (Decimal Representation)
class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=1024, out_sz=10, layers=[240,84]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz,layers[0])
        self.fc2 = nn.Linear(layers[0],layers[1])
        self.fc3 = nn.Linear(layers[1],out_sz)
    
    def forward(self,X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
    