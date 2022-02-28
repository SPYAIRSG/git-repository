import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import gzip
import csv
import matplotlib.pyplot as plt
import numpy as np

# Parameters
hidden_size = 100
batch_size = 256
n_layers = 2
n_epoch = 100
n_chars = 128
n_country = 18
USE_GPU = False


class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'data/names_train.csv.gz' if is_train_set else 'data/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]

    def __len__(self):
        return self.len

    def getCountryDict(self):
        # CountryDict = {self.country_list[i]: i for i in self.country_num}
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):
        return self.country_list[index]

    def getCountriesNum(self):
        return self.country_num

# prepare dataset
train_dataset = NameDataset(is_train_set=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataest = NameDataset(is_train_set=False)
test_loader = DataLoader(dataset=test_dataest, batch_size=batch_size, shuffle=False)

N_COUNTRY = train_dataset.getCountriesNum()

# design model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.emb = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        # input:[B, S]
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        embedding = self.emb(input)

        # pack them up
        gru_input = pack_padded_sequence(embedding, seq_lengths)

        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=-1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output

def name2list(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)
def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [s[0] for s in sequences_and_lengths]
    seq_lengths = torch.LongTensor([s[1] for s in sequences_and_lengths])
    countries = countries.long()

    # make tensor of name, [B, S]
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)



def train():
    total_loss = 0.0
    for i, (names, countries) in enumerate(train_loader):
        inputs, seq_lengths, target = make_tensors(names, countries)
        outputs = classifier(inputs, seq_lengths)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

def test():
    correct = 0
    total = len(test_dataest)
    print("evaluating trained model....")
    with torch.no_grad():
        for i, (names, countries) in enumerate(test_loader):
            inputs, seq_lengths, target = make_tensors(names, countries)
            outputs = classifier(inputs, seq_lengths)
            pred = outputs.max(dim=-1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        # percent = '%.2f' %(100*correct/total)
        # print(f'Test set: Accuracy {correct}/{total} {percent}%')

    return correct/total

if __name__ == '__main__':
    classifier = RNNClassifier(n_chars, hidden_size, n_country, n_layers)
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    print("Training for %d epochs..." % n_epoch)
    acc_list = []
    for epoch in range(1, n_epoch+1):
        total_loss = train()
        print("epoch %d, total loss: %.4f" % (epoch, total_loss))
        acc = test()
        print("epoch %d, acc: %.4f" % (epoch, acc))
        acc_list.append(acc)

    epoch_list = np.arange(1, n_epoch+1)
    plt.plot(epoch_list, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

