import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# prepare dataset
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        dataset = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = dataset.shape[0]
        self.x_data = torch.from_numpy(dataset[:, :-1])
        self.y_data = torch.from_numpy(dataset[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset("diabetes.csv.gz")
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4)


# design model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.activate = nn.Sigmoid()
    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x

model = Model()

# construct loss optimizer
criterion = nn.BCELoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Trainning Cycle
if __name__ == '__main__':

    for epoch in range(100):
        for i, data in enumerate(train_loader):
            inputs, label = data
            y_pred = model(inputs)
            loss = criterion(y_pred, label)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch + 1, loss.item())