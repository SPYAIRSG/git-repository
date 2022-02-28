import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 分类问题
# prepare dataset
dataset = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)

x_data = torch.from_numpy(dataset[:, :-1])
y_data = torch.from_numpy(dataset[:, [-1]])

# design model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 2)
        self.linear4 = nn.Linear(2, 1)
        self.activate = nn.Sigmoid()
    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = nn.Sigmoid()(self.linear4(x))
        return x

model = Model()

# construct loss and optimizer
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.1)


# trainning cycle

epoch_list = []
loss_list = []
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch+1, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_list.append(epoch)
    loss_list.append(loss.item())

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()