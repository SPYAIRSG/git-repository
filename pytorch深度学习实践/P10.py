import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

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
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(320, 10)
    def forward(self, x):
        # x:[batch_size, channel, H, W]
        batch_size = x.size(0)
        x = nn.ReLU()(self.pooling(self.conv1(x)))
        x = nn.ReLU()(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

model = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# construct loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Trainning cycle
def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader):
        inputs, label = data
        inputs, label = inputs.to(device), label.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index % 300 == 299:
            print("[%d %5d] loss: %.3f" %(epoch+1, batch_index+1, running_loss/300))


def test():
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, label = data
            inputs, label = inputs.to(device), label.to(device)
            outputs = model(inputs)
            _, predict = torch.max(outputs, dim=-1)
            total += label.size(0)
            correct += (predict == label).sum().item()
    print("test acc:", 100*correct/total)


if __name__ == '__main__':
    for epoch in range(10):

        train(epoch)
        test()
        print("-----------------------------")