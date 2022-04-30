import math
from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from numpy import float32
from Data import getAttackDataIterator, getNormalDataIterator, SequencedDataIterator

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, seq_len = 10):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
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
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
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

def train_model(model, optim, epochs, train_iter, print_every=128):
    
    model.train()
    
    start = time.time()
    temp = start
    
    total_loss = 0
    
    for epoch in range(epochs):
       
        for i, batch in enumerate(train_iter):
            src = torch.tensor(batch[:, :sequenceLength, :].astype(float32)).float()
            trg = torch.tensor(batch[:, sequenceLength, :].astype(float32)).float()

            if torch.cuda.is_available():
                src = src.cuda()
                trg = trg.cuda()

            preds = model(src)
            
            optim.zero_grad()
            
            loss = F.mse_loss(preds, trg)
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


if __name__ == '__main__':

    d_model = 51
    heads = 17
    embed_dim = 51
    sequenceLength = 10
    train_batchSize = 512
    test_batchSize = 16384
    normal_iter = getNormalDataIterator(train_batchSize, sequenceLength + 1, True)
    model = Generator(d_model=d_model, seq_len=sequenceLength, embed_dim=embed_dim, num_heads=heads)
    if torch.cuda.is_available():
        model = model.to('cuda')#model.cuda()
    lr = 0.05
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.float()

    train_model(model=model, optim=optim, epochs=5, train_iter=normal_iter, print_every=128)
