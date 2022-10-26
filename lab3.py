from statistics import mode
from turtle import forward
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


x_train = torch.rand(500)

y_train = torch.zeros(500)
for i in range(100):
    y_train[100+i]+=1
#y_train = (y_train>0.5).float()

x_valid = torch.rand(500)


x_valid.unsqueeze_(1)
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

plt.plot(x_train.detach().numpy(),y_train.detach().numpy(),'.')
plt.show()

class optimalNet(nn.Module):
    def __init__(self,n_hid_n):
        super(optimalNet,self).__init__()
        self.fc1 = nn.Linear(1,n_hid_n)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(n_hid_n,n_hid_n)
        self.act2 = nn.Sigmoid()
        self.fc4 = nn.Linear(n_hid_n,1)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc4(x)
        return x


optimalNet = optimalNet(10000)

def predict(net,x,y):
    y_pred = net.forward(x)
    #y_pred = (y_pred>0.5).float()
    plt.plot(x.detach().numpy(),y[:500].detach().numpy(),'.',c = 'g')
    plt.plot(x.detach().numpy(),y_pred.detach().numpy(),'.',c = 'r')
    plt.show()

predict(optimalNet,x_valid,y_train)

optimiser = torch.optim.Adam(optimalNet.parameters(),lr = 0.0001)

def loss(pred,true):
    sq = (pred - true)**2
    return sq.mean()


for e in range(100):
    optimiser.zero_grad()

    y_pred = optimalNet.forward(x_train)
    loss_val  = loss(y_pred,y_train)

    print(loss_val)

    loss_val.backward()
    optimiser.step()

predict(optimalNet,x_valid,y_train)
