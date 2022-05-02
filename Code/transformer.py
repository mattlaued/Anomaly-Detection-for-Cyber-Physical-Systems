import math
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from numpy import float32
from Data import getNormalDataIterator, getNormalData

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, seq_len = 10):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        pe = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model - 1, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        var = None
        var = Variable(self.pe[:,:self.seq_len], requires_grad=False)
        x = x + var
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, seq_len, embed_dim, num_heads) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.pos_encoding = PositionalEncoder(d_model=d_model, seq_len=seq_len)
        self.norm1 = nn.LayerNorm((seq_len, 51))
        self.norm2 = nn.LayerNorm(510)
        self.fwd = nn.Linear(in_features=510, out_features=510)

    def forward(self, x):
        pe = self.pos_encoding(x)
        res, weights = self.attention(pe, pe, pe)
        normed = self.norm1(x + res)
        flat = torch.flatten(normed, start_dim=1)
        lin = self.fwd(flat)
        added = lin + flat
        ret = self.norm2(added)
        return ret

class Generator(nn.Module):
    def __init__(self, d_model, seq_len, embed_dim, num_heads) -> None:
        super().__init__()
        self.transformer = Transformer(d_model, seq_len, embed_dim, num_heads)
        self.lin1 = nn.Linear(in_features=510, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=32)
        self.lin3 = nn.Linear(in_features=32, out_features=51)
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x

def train_model(model, optim, seq_len, epochs, train_iter, test_data=None, chkpts_dir=None, print_every=128):

    model.float()
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.train()
    
    start = time.time()
    temp = start
    
    total_loss = 0
    
    for epoch in range(epochs):
       
        for i, batch in enumerate(train_iter):
            src = torch.tensor(batch[:, :seq_len, :].astype(float32)).float()
            trg = torch.tensor(batch[:, seq_len, :].astype(float32)).float()

            if torch.cuda.is_available():
                src = src.cuda()
                trg = trg.cuda()

            preds = model(src)
            
            optim.zero_grad()
            
            loss = F.mse_loss(preds, trg)
            if not math.isnan(loss):
                loss.backward()
                optim.step()
                
                total_loss += loss.data.item()
                if (i + 1) % print_every == 0:
                    loss_avg = total_loss / print_every
                    print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60,
                            epoch + 1, i + 1, loss_avg, time.time() - temp,
                            print_every))
                    total_loss = 0
                    temp = time.time()
            else:
                print(f'nan loss, discarding batch. epoch {epoch} iteration {i}')

        print(f'epoch {epoch + 1} complete.')
        if test_data is not None:
            print('testing data...')
            loss = 0
            norm = 0
            length = 0
            for i in range(0, len(test_data) // (seq_len + 1), seq_len + 1):
                src = torch.tensor(test_data[i:i + seq_len].astype(float32))
                trg = torch.tensor(test_data[i + seq_len].astype(float32))
                if torch.cuda.is_available():
                    src = src.cuda()
                    trg = trg.cuda()
                
                preds = model(src)

                loss += F.mse_loss(preds, trg).data.item()
                norm += torch.norm(trg - preds).data.item()
                length += 1

            loss /= length
            norm /= length

            print(f'epoch {epoch} complete. average test loss: {loss}, average norm of diff: {norm}')

            if chkpts_dir is not None:
                filepath = chkpts_dir + f'/Checkpoints/transformer/generator_e{epochs}lr001_1_{epoch + 1}.pt'
                torch.save(model.state_dict(), filepath)
                print(f'epoch saved to {filepath}')


if __name__ == '__main__':
    d_model = 51
    heads = 17
    embed_dim = 51
    sequenceLength = 10
    train_batchSize = 256
    test_batchSize = 80000
    epochs = 8
    normal_iter = getNormalDataIterator(train_batchSize, sequenceLength + 1, True)
    test_data = getNormalData()[1:test_batchSize+1]
    model = Generator(d_model=d_model, seq_len=sequenceLength, embed_dim=embed_dim, num_heads=heads)
    lr = 0.001
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    chkpts_dir = os.path.abspath(os.getcwd())
    final_chkpts_dir = chkpts_dir + f'/Checkpoints/transformer/generator_e{epochs}lr001_1_final.pt'

    train_model(model=model, optim=optim, seq_len=sequenceLength, epochs=epochs, train_iter=normal_iter, test_data=test_data, chkpts_dir=chkpts_dir, print_every=128)
    torch.save(model.state_dict(), chkpts_dir)
    print(f'Training complete, model saved to {final_chkpts_dir}')
