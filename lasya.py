import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# рандомим числа (1000 массивов по 3 числа от 0 до 1): степень принадлежности
x_yes = np.random.rand(1000, 3)

# больше 0,5 - 1: класс ВМФ
y_yes = torch.zeros(1000)
for i in range(100):
    y_yes[300+i] = 1
    y_yes[900+i] = 1
#y_yes = (y_yes > 0.5).float()

# для обучения (взяли 800)
x_train = torch.FloatTensor(x_yes)[:800]
y_train = y_yes[:800]
# для тестирования (взяли 200)
x_test = torch.FloatTensor(x_yes)[800:-1]
y_test = y_yes[800:-1]

# вид в виде столба
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_test.unsqueeze_(1)
y_test.unsqueeze_(1)

x = torch.range(0,len(y_yes)-1)
plt.plot(x, y_yes.reshape(-1).detach().numpy(), '.')
plt.show()

# применение линейной функции
class optimalNet(nn.Module):
    def __init__(self, n_hid_n):
        super(optimalNet, self).__init__()
        # 3 хар-ки приводятся в 1000 нейронов
        self.fc1 = nn.Linear(3, n_hid_n)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.fc3 = nn.Linear(n_hid_n, 1)
        self.act3 = nn.ReLU()
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)


        return x

optimalNet = optimalNet(10)

def predict(net, x, y):
    y_pred = net.forward(x)
    x = torch.range(0,len(y)-1)
    plt.plot(x,y_pred.reshape(-1).detach().numpy(), '.', c = 'r')
    plt.plot(x,y.reshape(-1).detach().numpy(), '.', c = 'g')
    plt.show()



predict(optimalNet, x_test, y_test)

#вычисление ошибки
def loss(pred, true):
    error = (pred-true)**2
    return error.mean()

# шаг
optimiser = torch.optim.Adam(optimalNet.parameters(), lr = 0.001)

# с помощью градиентного спуска нейронная сеть обновляет веса в нейронах для лучшего определения класса корабля
for i in range(5000):
    optimiser.zero_grad()

    y_pred = optimalNet.forward(x_train)
    loss_val = loss(y_pred, y_train)

    print(loss_val)

    loss_val.backward()
    optimiser.step()

predict(optimalNet, x_test, y_test)

predict(optimalNet, x_train, y_train)