import torch
from torch import nn
import matplotlib.pyplot as plt

# генерация данных
x_ = [i / 1000 for i in range(1000)]
y_ = torch.zeros(1000)
for i in range(200):
    y_[40 + i] = 1
# y_ = (y_ > 0.5).float()

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


class optimalNet(nn.Module):
    def __init__(self, n_hid_n):
        super(optimalNet, self).__init__()
        self.fc1 = nn.Linear(1, n_hid_n)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(n_hid_n, n_hid_n)
        self.act2 = nn.Sigmoid()
        self.fc3 = nn.Linear(n_hid_n, n_hid_n)
        self.act3 = nn.Sigmoid()
        self.fc4 = nn.Linear(n_hid_n, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        return x


optimalNet = optimalNet(500)


def predict(net, x, y):
    y_pred = net.forward(x)
    # y_pred = (y_pred>0.5).float()
    plt.plot(x.detach().numpy(), y.detach().numpy(), '.', c='g')
    plt.plot(x.detach().numpy(), y_pred.detach().numpy(), '.', c='r')
    plt.show()


predict(optimalNet, x_test, y_test)

optimiser = torch.optim.Adam(optimalNet.parameters(), lr=0.0001)


def loss(pred, true):
    sq = (pred - true) ** 2
    return sq.mean()


for e in range(500):
    optimiser.zero_grad()

    y_pred = optimalNet.forward(x_train)
    loss_val = loss(y_pred, y_train)

    print(loss_val)

    loss_val.backward()
    optimiser.step()

predict(optimalNet, x_test, y_test)