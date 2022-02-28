import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


# y = np.array([1, 0, 0])
# z = np.array([0.2, 0.1, -0.1])
#
# y_pred = np.exp(z) / np.exp(z).sum()
# loss = (-y * np.log(y_pred)).sum()
# print(loss)

# criterion = nn.CrossEntropyLoss()
# Y = torch.LongTensor([2, 0, 1])
#
# Y_pred1 = torch.tensor([[0.1, 0.2, 0.9],
#                         [1.1, 0.1, 0.2],
#                         [0.2, 2.1, 0.1]])
#
# Y_pred2 = torch.tensor([[0.8, 0.2, 0.3],
#                         [0.2, 0.3, 0.5],
#                         [0.2, 0.2, 0.5]])
#
#
# loss1 = criterion(Y_pred1, Y)
# loss2 = criterion(Y_pred2, Y)
#
# print("loss1:", loss1.item())
# print("loss2:", loss2.item())

# prepare dataset
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               transform=transform,
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              transform=transform,
                              download=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# design model
class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 10)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.activate(self.linear4(x))
        x = self.linear5(x)
        return x

model = Classification()

# construct loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader):
        inputs, label = data
        outputs = model(inputs)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_index % 300 == 299:
            print("[%d, %5d] loss: %.3f" %(epoch+1, batch_index+1, running_loss/300))
            running_loss = 0.0
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, label = data
            outputs = model(inputs)
            _, predict = torch.max(outputs, dim=-1)    # [value, index]
            total += label.size(0)
            correct += (predict == label).sum().item()
    print("acc:", 100*correct/total)


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        print("---------------------------------------")
        test()
