import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

x_ = [i / 1000 for i in range(1000)]
y_ = torch.zeros(1000)

for i in range(200):
    y_[40 + i] = 1



# для обучения
x_train = torch.FloatTensor(x_)[:800]
y_train = y_[:800]

# для тестирования
x_test = torch.FloatTensor(x_)[800:-1]
y_test = y_[800:-1]

x_test.unsqueeze_(1)
y_test.unsqueeze_(1)
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

plt.plot(x_train.detach().numpy(), y_train.detach().numpy(), '.')
plt.show()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



model = NeuralNetwork()

def predict(net, x, y):
    y_pred = net(x)
    plt.plot(x.detach().numpy(), y.detach().numpy(), '.', c='g')
    plt.plot(x.detach().numpy(), y_pred.detach().numpy(), '.', c='r')
    plt.show()

predict(model, x_test, y_test)



optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def loss(pred, true):
    sq = (pred - true) ** 2
    return sq.mean()

def train(x,y, model, optimizer):
    model.train()
    for epoch in range(5000):

        pred = model(x)
        loss_val = loss(pred, y)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


        print(f"loss: {loss_val}")

train(x_train,y_train,model,optimizer)

predict(model, x_test, y_test)

predict(model, x_train, y_train)


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")