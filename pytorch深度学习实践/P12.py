import torch
import torch.nn as nn
import torch.optim as optim


# 1. RNNCell
def RNNCell():
    batch_size = 1
    seq_len = 3
    input_size = 4
    hidden_size = 2

    cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

    # (seq, batch_size, features)
    dataset = torch.randn(seq_len, batch_size, input_size)
    hidden = torch.zeros(batch_size, hidden_size)

    for idx, input in enumerate(dataset):
        print("-"*20, idx, "-"*20)
        print("Input size:", input.shape)    # [batch_size, input_size]
        hidden = cell(input, hidden)

        print("output size:", hidden.shape)
        print(hidden)


# 2. RNN
def RNN():
    batch_size = 1
    seq_len = 3
    input_size = 4
    hidden_size = 2
    num_layers = 1

    # construct RNN
    rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    # (seq, batch_size, features)
    inputs = torch.randn(seq_len, batch_size, input_size)
    hidden = torch.zeros(num_layers, batch_size, hidden_size)

    outputs, hidden = rnn(inputs, hidden)

    print("outputs size:", outputs.size())
    print("outputs:", outputs)
    print("Hidden size:", hidden.size())
    print("Hidden:", hidden)

# 3. RNNCell Model - NLP
def RNNCell_NLP():

    batch_size = 1
    input_size = 4
    hidden_size = 4

    idx2char = ['e', 'h', 'l', 'o']
    x_data = [1, 0, 2, 2, 3]       # hello
    y_data = [3, 1, 2, 3, 2]       # ohlol

    one_hot_lookup = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

    x_one_hot = [one_hot_lookup[x] for x in x_data]

    inputs = torch.tensor(x_one_hot, dtype=torch.float32).view(-1, batch_size, input_size)    # [seq_len, batch_size, input_size]
    labels = torch.LongTensor(y_data).view(-1, 1)        # [seq_len, 1]

    class Model(nn.Module):
        def __init__(self, input_size, batch_size, hidden_size):
            super(Model, self).__init__()
            self.input_size = input_size
            self.batch_size = batch_size
            self.hidden_size = hidden_size

            self.rnncell = nn.RNNCell(input_size=self.input_size,
                                      hidden_size=self.hidden_size)


        def forward(self, input, hidden):
            hidden = self.rnncell(input, hidden)
            return hidden

        def init_hidden(self):
            return torch.zeros(self.batch_size, self.hidden_size)

    net = Model(input_size, batch_size, hidden_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.1)

    for epoch in range(15):
        loss = 0
        optimizer.zero_grad()
        hidden = net.init_hidden()
        print('predicted String:', end='')
        for input, label in zip(inputs, labels):
            hidden = net(input, hidden)
            loss += criterion(hidden, label)
            _, idx = hidden.max(dim=1)
            print(idx2char[idx.item()], end='')
        loss.backward()
        optimizer.step()
        print(', Epoch [%d/15] loss=%.4f' %(epoch+1, loss.item()))

# 4. RNN Model - NLP
def RNN_NLP():
    batch_size = 1
    input_size = 4
    hidden_size = 4
    num_layers = 1
    seq_len = 5

    idx2char = ['e', 'h', 'l', 'o']
    x_data = [1, 0, 2, 2, 3]  # hello
    y_data = [3, 1, 2, 3, 2]  # ohlol

    one_hot_lookup = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

    x_one_hot = [one_hot_lookup[x] for x in x_data]

    inputs = torch.tensor(x_one_hot, dtype=torch.float32).view(-1, batch_size, input_size)  # [seq_len, batch_size, input_size]
    labels = torch.LongTensor(y_data)  # [seq_len*batch, 1]

    class Model(nn.Module):
        def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
            super(Model, self).__init__()
            self.num_layers = num_layers
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.batch_size = batch_size

            self.rnn = nn.RNN(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers)

        def forward(self, input):
            hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

            output, _ = self.rnn(input, hidden)    # output:[seq_len, batch_size, hidden_size]
            return output.view(-1, self.hidden_size)


    net = Model(input_size, hidden_size, batch_size, num_layers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.1)

    for epoch in range(15):

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, idx = outputs.max(dim=-1)
        idx = idx.data.numpy()
        print('predicted:', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))

def Emb():
    input_size = 4
    embedding_size = 10
    hidden_size = 8
    num_layers = 2
    num_class = 4
    batch_size = 1
    seq_len = 5

    idx2char = ['e', 'h', 'l', 'o']
    x_data = [[1, 0, 2, 2, 3]]   # [batch_size, seq_len]
    y_data = [3, 1, 2, 3, 2]     # [batch_size*seq]

    inputs = torch.LongTensor(x_data)
    labels = torch.LongTensor(y_data)


    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.emb = nn.Embedding(input_size, embedding_size)
            self.rnn = nn.RNN(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
            self.fc = nn.Linear(hidden_size, num_class)

        def forward(self, x):
            hidden = torch.zeros(num_layers, x.size(0), hidden_size)
            x = self.emb(x)
            x, _ = self.rnn(x, hidden)
            x = self.fc(x)
            return x.view(-1, num_class)

    net = Model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.05)

    for epoch in range(15):

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, idx = outputs.max(dim=-1)
        idx = idx.data.numpy()
        print('predicted:', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))


if __name__ == '__main__':

    # RNNCell()
    # RNN()
    # RNNCell_NLP()
    # RNN_NLP()
    Emb()