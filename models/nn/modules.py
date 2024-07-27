import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class cba(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class cbn(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))

class residual_conv(nn.Module):
    def __init__(self, cin, cout, stride=1, expansion=1):
        super().__init__()

        self.expansion = expansion
        self.downsample = None

        self.in_conv = cba(cin, cout, 3, stride, 1, bias=False)
        self.out_conv = cbn(cout, cout*self.expansion, 3, stride=1, padding=1, bias=False)

        self.out_act = nn.ReLU(inplace=True)

        if stride != 1:
            self.downsample = cbn(cin, cout*self.expansion, 1, stride, bias=False)

    def forward(self, x):
        y = self.in_conv(x)
        y = self.out_conv(y)

        if self.downsample is not None:
            x = self.downsample(x)
        y = y + x
        y = self.out_act(y)
        return y

class residual_layer(nn.Module):
    def __init__(self, cin, cout, stride=1, expansion=1, repeat=1):
        super().__init__()

        layers = []
        layers.append(residual_conv(cin, cout, stride, expansion))
        self.mid = cout * expansion

        for i in range(1, repeat):
            layers.append(residual_conv(self.mid, cout, expansion))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


#inspired by : https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3*embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.xavier_uniform()

    def xavier_uniform(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        # if bias
        # self.qkv.bias.data.fill_(0)
        ##self.out_proj.bias.data.fill_(0)

    def forward(self, x):
        b, l, d = x.size()
        qkv = self.qkv(x)

        qkv = qkv.reshape(b, l, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0,2,1,3) #Batch, Head, SeqLen, Dims
        q, k, v = qkv.chunk(3, dim=-1)

        ### Scaled dot product
        attn = (q @ k.transpose(-2, -1))
        attn = attn / math.sqrt(q.size()[-1])
        attn = F.softmax(attn, dim=-1)
        values = torch.matmul(attn, v)
        ### End

        values = values.permute(0, 2, 1, 3) #B, S, H, D

        #print(nn.CosineSimilarity()(values[:,:,0,:], values[:,:,1,:])) maybe like this ??

        values = values.reshape(b, l, self.embed_dim)
        outputs = self.out_proj(values)



        return outputs


#Encoder from Attention is all you need
class classic_encoder(nn.Module):
    def __init__(self, input_dim, num_heads, dim_ffn, dropout=0.0):
        super().__init__()

        self.mha = MultiHeadAttention(input_dim, input_dim, num_heads)


        #Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_ffn),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ffn, input_dim)
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn = self.mha(x)
        x = x + self.dropout(attn)
        x = self.norm1(x)

        y = self.linear_net(x)
        x = x + self.dropout(y)
        x = self.norm2(x)

        return x


class vit_encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
         super().__init__()

         self.layernorm1 = nn.LayerNorm(embed_dim)
         self.layernorm2 = nn.LayerNorm(embed_dim)

         self.attn = MultiHeadAttention(embed_dim, embed_dim, num_heads)

         self.linear = nn.Sequential(
             nn.Linear(embed_dim, hidden_dim),
             nn.GELU(),
             nn.Dropout(dropout),
             nn.Linear(hidden_dim, embed_dim),
             nn.Dropout(dropout)
         )

    #Z'l = MHA(LN(Zl-1)) + Zl-1
    #MLP(LN(Z'l)) + Z'l
    def forward(self, x):
        qkv = self.layernorm1(x)
        x = x + self.attn(qkv)
        x = x + self.linear(self.layernorm2(x))
        return x



