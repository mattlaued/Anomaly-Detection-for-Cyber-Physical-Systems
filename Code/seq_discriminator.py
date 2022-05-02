import time
import math
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from Data import getAttackDataIterator, getAttackData
from transformer import Generator
from tqdm import tqdm

def get_diffs_train(model, iter):
    '''
    Returns diffs between preds and actual, and labels
    '''
    diffs = []
    labels = []
    print('generating train diffs...')
    with tqdm(total=len(iter)) as prog:
        for i, (data, label) in enumerate(iter):
            if i > 3521:
                break
            src = torch.tensor(data[:, :-1], dtype=torch.float32)
            trg = torch.tensor(data[:, -1], dtype=torch.float32)
            pred = model(src)
            diffs.append(torch.abs(pred - trg))
            labels.append(torch.tensor([label], dtype=torch.float32))
            prog.update()
    print('train diffs generation complete')
    return torch.stack(diffs), torch.stack(labels).reshape((len(diffs), diffs[0].shape[0], 1))

def get_diffs_test(model, data, labels, seq_len):
    diffs = []
    label = []
    print('generating test diffs...')
    with tqdm(total=data.shape[0] - seq_len) as prog:
        for i in range(0, data.shape[0] - seq_len, seq_len + 1):
            src = torch.tensor(data[i:i + seq_len], dtype=torch.float32)
            trg = torch.tensor(data[i + seq_len], dtype=torch.float32)
            pred = model(src)
            diffs.append(torch.abs(pred - trg))
            label.append(labels[i + seq_len])
            prog.update(seq_len + 1)
    print('test diffs generation complete')
    return torch.stack(diffs), torch.tensor(label, dtype=torch.float32)

class Discriminator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(51,128)
        self.lin2 = nn.Linear(128, 256)
        self.lin3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.sigmoid(self.lin3(x))
        return x

def train_model(model, optim, epochs, train_data, train_labels, test_data, test_labels, chkpts_dir=None):
    model.float()
    model.train()

    start = time.time()
    temp = start

    total_loss = 0

    for epoch in range(epochs):
        for i in range(train_data.shape[0]):
            preds = model(train_data[i])
            optim.zero_grad()
            loss = F.binary_cross_entropy(preds, train_labels[i])
            if not math.isnan(loss):
                loss.backward()
                optim.step()
                total_loss += loss.data.item()
                if (i + 1) % 128 == 0:
                    loss_avg = total_loss / 128
                    print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60,
                            epoch + 1, i + 1, loss_avg, time.time() - temp, 128))
                    total_loss = 0
                    temp = time.time()
            else:
                print(f'nan loss, discarding batch. epoch {epoch} iteration {i}')
        print(f'epoch {epoch + 1} complte. testing data...')
        preds = model(test_data).reshape(test_labels.shape)
        loss = F.binary_cross_entropy(preds, test_labels)
        acc = torch.sum(test_labels == preds).item() / test_labels.size(0)
        print(f'epoch {epoch + 1}: loss {loss}, accuracy {acc}')
        if chkpts_dir is not None:
            filepath = chkpts_dir + f'disc/seq_disc_e{epochs}lr001_1_{epoch + 1}.pt'
            torch.save(model.state_dict(), filepath)
            print(f'epoch saved to {filepath}')

if __name__ == '__main__':
    d_model = 51
    heads = 17
    embed_dim = 51
    train_batchSize = 128
    test_batchSize = 40000
    seq_len = 10
    EPOCHS = 10
    lr = 0.001
    chkpts_dir = os.path.abspath(os.getcwd()) + '/Checkpoints/transformer/'

    generator = Generator(d_model, seq_len, embed_dim, heads)
    
    generator.load_state_dict(torch.load(chkpts_dir + 'generator_e8lr001_1_7.pt'))
    for param in generator.parameters():
        param.requires_grad = False

    train_iter = getAttackDataIterator(batchSize=train_batchSize, sequenceLength=seq_len+1, includeData=True, includeLabel=True)
    train_data, train_labels = get_diffs_train(generator, train_iter)
    del train_iter
    test_data, test_labels = getAttackData()
    test_data = test_data[1:test_batchSize + 1]
    test_labels = test_labels[1:test_batchSize + 1]
    test_data, test_labels = get_diffs_test(generator, test_data, test_labels, seq_len)
    
    disc = Discriminator()
    optim = torch.optim.SGD(disc.parameters(), lr=lr)
    train_model(disc, optim, EPOCHS, train_data, train_labels, test_data, test_labels, chkpts_dir)
    print('training complete')
    quit()
