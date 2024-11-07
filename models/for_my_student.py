import torch
import torch.nn as nn


class my_linear(nn.Module):
    def __init__(self, hidden_dim, out=10):
        super().__init__()
        
        self.l = nn.Linear(3, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.l3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        self.out = nn.Linear(hidden_dim*32*32*4, out)
        
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0,2,3,1)
        h1 = self.act(self.l(x))
        h2 = self.act(self.l2(h1))
        h3 = self.act(self.l3(h2))
        return self.out(h3.flatten(2,3).flatten(1,2))
    
    
class my_cnn(nn.Module):
    def __init__(self, hidden_dim, out=10):
        super().__init__()
        
        self.mp = nn.MaxPool2d((2,2))
        self.c1 = nn.Conv2d(3, hidden_dim, 3, 1, 1)
        self.c2 = nn.Conv2d(hidden_dim, hidden_dim*2, 3, 1, 1)
        self.c3 = nn.Conv2d(hidden_dim*2, hidden_dim*6, 3, 1, 1)
        #self.c4 = nn.Conv2d(hidden_dim, out, 3, 1, 1)
        self.lr = nn.Linear(4*4*hidden_dim*6, out)
        self.act = nn.ReLU()
        
    def forward(self, x):
        h1 = self.mp(self.act(self.c1(x)))
        h2 = self.mp(self.act(self.c2(h1)))
        h3 = self.mp(self.act(self.c3(h2)))
        h4 = self.lr(h3.flatten(2,3).flatten(1,2))
        return h4
