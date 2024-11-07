import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import pandas as pd

from einops import repeat



#I want to check if from the begining, squeeze and excite just tend to add more similarity to the tensor!
class squeeze_and_excite(nn.Module):
    def __init__(self, c_in, reduction=16):
        super().__init__()
        c_mid = math.ceil(c_in / reduction)

        self.l_in = nn.Sequential(nn.Linear(c_in, c_mid, bias=True), nn.ReLU())
        self.l_out = nn.Sequential(nn.Linear(c_mid, c_in, bias=True), nn.Sigmoid())


    def forward(self,x):     
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0,2,3,1)
        y = self.l_in(y)
        y = self.l_out(y)
        y = y.permute(0,3,1,2)
        return x * y


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



class shufflenet_v2_block(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()

        branch_features = c_out // 2
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, stride, 1, groups=c_in, bias=False), #seems to be false inside PyTorch ShuffleNetV2 implementation.
                nn.BatchNorm2d(c_in),
                cba(c_in, branch_features, 1, 1, 0, bias=False),
            )
        else:
            self.branch1 = nn.Indentity()

        self.branch2 = nn.Sequential(
            cba(c_in if (stride > 1) else branch_features, branch_features, 1, 1, 0, bias=False),
            cbn(branch_features, branch_features, 3, stride, 1, groups=branch_features),
            cba(branch_features, branch_features, 1, 1, 0, bias=False),
        )

        self.stride = stride
        
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        
        #channel shuffle here, but I don't care for my research (at least for now)
        #TODO : Channel Shuffle :)

        return out


class residual_conv(nn.Module):
    def __init__(self, cin, cout, stride=1, expansion=1):
        super().__init__()

        self.expansion = expansion
        self.downsample = None

        self.in_conv = cba(cin, cout, 3, stride, 1, bias=False)
        self.out_conv = cbn(cout, cout*self.expansion, 3, stride=1, padding=1, bias=False)

        self.se = squeeze_and_excite(cout*self.expansion)

        self.out_act = nn.ReLU(inplace=True)

        if stride != 1:
            self.downsample = cbn(cin, cout*self.expansion, 1, stride, bias=False)

    def forward(self, x, print_cos_cs=False):
        cos = None
        y = self.in_conv(x)
        y = self.out_conv(y)

        if self.downsample is not None:
            x = self.downsample(x)

        if print_cos_cs:
            cos = cos_sim_in(y.flatten(-2,-1))

        y = self.se(y)#right before residuals...
        
        if print_cos_cs:
            cos = cos - cos_sim_in(y.flatten(-2,-1))

        y = y + x
        y = self.out_act(y)

        if print_cos_cs:
            return y, cos
        return y

class residual_layer(nn.Module):
    def __init__(self, cin, cout, stride=1, expansion=1, repeat=1):
        super().__init__()

        layers = []
        layers.append(residual_conv(cin, cout, stride, expansion))
        self.mid = cout * expansion

        for i in range(1, repeat):
            layers.append(residual_conv(self.mid, cout, expansion))

        self.layers = nn.ModuleList(layers)

    def forward(self, x, print_cos_cs=False):
        total_cos = []
        for hidden_layer in self.layers:
            x = hidden_layer(x, print_cos_cs)
            if isinstance(x, tuple):
                x, cos = x
                total_cos.append(cos)
    
        if print_cos_cs:
            return x, total_cos
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'



#rewrite the star ops ! 

class star_block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.dw_conv = cbn(dim, dim, 7, 1, (7-1)//2, groups=dim)

        #self.f = nn.Conv2d(dim, int(dim*6.), 1)
        self.f1 = nn.Conv2d(dim, int(dim*mlp_ratio), 1)
        self.f2 = nn.Conv2d(dim, int(dim*mlp_ratio), 1)
        #self.g = cbn(int(6.*dim), dim, 1)
        self.g = cbn(int(mlp_ratio*dim), dim, 1)

        self.dw_conv_2 = nn.Conv2d(dim, dim, 7, 1, (7-1)//2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dw_conv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        """x = self.f(x)
        if self.train:
            #x = self.act(x)
            x = self.act(x)*x"""
        x = self.dw_conv_2(self.g(x))
        x = identity + self.drop_path(x)
        return x


class square_block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.dw_conv = cbn(dim, dim, 7, 1, (7-1)//2, groups=dim)

        self.f = nn.Conv2d(dim, int(dim*(mlp_ratio * 2)), 1)

        self.g = cbn(int((mlp_ratio * 2)*dim), dim, 1)


        self.dw_conv_2 = nn.Conv2d(dim, dim, 7, 1, (7-1)//2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dw_conv(x)
        x = self.f(x)
        x = self.act(x) * F.max_pool2d(x, kernel_size=5, stride=1, padding=5//2) #est-ce que ça va marcher ?
        
        x = self.dw_conv_2(self.g(x))
        x = identity + self.drop_path(x)
        return x



class star2_block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.dw_conv = cbn(dim, dim, 7, 1, (7-1)//2, groups=dim)
        #pas d'activation : en théorie, s'il n'y pas d'activation en chain rule : f(g(x)) complexifie juste g avec f
        #l'activation permet une indépendance des 2 : f(t(g(x)))

        self.f1 = nn.Conv2d(dim, int(dim*mlp_ratio), 1)
        self.f2 = nn.Conv2d(dim, int(dim*mlp_ratio), 1)

        #techniquement si : f1(x) * f2(x) = f1(x)² + f1(x) * (f2(x) - f1(x))
        #et que f(x) dans le forme square f(x)² est équivalent voir > avec moins de paramètre
        #on peut faire l'équivalence avec f1(x) dim = 6*dim (au lieu de 8) et f2(x) = 2*dim 
        #(pour voir s'il existe de quoi ajouter)
        
        self.g = cbn(int(mlp_ratio*dim), dim, 1)

        self.dw_conv_2 = nn.Conv2d(dim, dim, 7, 1, (7-1)//2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dw_conv(x)

        x1, x2 = self.f1(x), self.f2(x)
        xsquare = self.act(x1) * x1
        epsilon = (x2-x1)
        x = xsquare + x1 * epsilon #should be exactly the same thing as x1*x2 
        #(might not be the case since the activation right at calculating xsquare exhibit some variations)

        x = self.dw_conv_2(self.g(x))
        x = identity + self.drop_path(x)
        return x



class RecurrentInvertedBottleneck(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.dw_conv = cbn(dim, dim, 7, 1, (7-1)//2, groups=dim)

        self.W = nn.Parameter(torch.randn(dim, int(dim*mlp_ratio)))
        self.b = nn.Parameter(torch.randn(dim))
        self.dw_conv_2 = nn.Conv2d(dim, dim, 7, 1, (7-1)//2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dw_conv(x)

        x = ( x.permute(0,2,3,1) @ self.W )
        x = self.act(x) * x
        x = ( x @ self.W.T ).permute(0,3,1,2)

        x = self.dw_conv_2(x)
        x = identity + self.drop_path(x)
        return x
        



class star2_t_block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.dw_conv = cbn(dim, dim, 7, 1, (7-1)//2, groups=dim)
        #pas d'activation : en théorie, s'il n'y pas d'activation en chain rule : f(g(x)) complexifie juste g avec f
        #l'activation permet une indépendance des 2 : f(t(g(x)))

        self.f1 = nn.Conv2d(dim, int(dim*mlp_ratio), 1)
        self.f2 = nn.Conv2d(dim, int(dim), 1)

        #techniquement si : f1(x) * f2(x) = f1(x)² + f1(x) * (f2(x) - f1(x))
        #et que f(x) dans le forme square f(x)² est équivalent voir > avec moins de paramètre
        #on peut faire l'équivalence avec f1(x) dim = 6*dim (au lieu de 8) et f2(x) = 2*dim 
        #(pour voir s'il existe de quoi ajouter)


        #ici je teste cette forme : 
        # f(x)² + x * (g(x)-x)
        
        self.g = cbn(int(mlp_ratio*dim), dim, 1)

        self.dw_conv_2 = nn.Conv2d(dim, dim, 7, 1, (7-1)//2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dw_conv(x)

        x1, x2 = self.f1(x), self.f2(x)
        xsquare = self.act(x1) * x1
        #epsilon = (x2-x1)
        x = xsquare + x * (x2-x) #should be exactly the same thing as x1*x2 
        #(might not be the case since the activation right at calculating xsquare exhibit some variations)

        x = self.dw_conv_2(self.g(x))
        x = identity + self.drop_path(x)
        return x











def cosine_similarity(x):
    # x inf the form of Batch, Head, SeqLen, Dims (we want to check similarity in Dims-wise dimension of each heads one to anthoer (i,j))
    

    b,h,s,d = x.shape
    a=nn.CosineSimilarity(dim=2, eps=1e-6) #on head dimmension

    mean = []
    for i in range(x.shape[1]):
        for j in range(x.shape[1]): #for each heads (compared to another)
            if i != j: #not the SAME one !
                #if equals, sum() == all_dim (but the one select) so here : Batch, (not head!) seq, dim 
                #unsqueeze to keep dim at 1 ( cosinesimilarity happy!:) )
                mean.append(a(x[:,i,...].unsqueeze(1), x[:,j,...].unsqueeze(1)).sum())  #sum of each generated tensor
    
    t = torch.tensor(mean)
    t = (t - t.min()) / (t.max() - t.min()) #normalization
    
    #print(t.mean())
    
    return t.mean() #mean

    
  
    

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

    def forward(self, x, print_opt_cs=False):
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
        
        cos = None
        if print_opt_cs:
            cos = pw_cos_sim(values)
            #print("Cosine similarity of V after (Q@K.t)@V - mean = ", cosine_similarity(values))
        
        values = values.permute(0, 2, 1, 3) #B, S, H, D

        #print(nn.CosineSimilarity()(values[:,:,0,:], values[:,:,1,:])) maybe like this ??
    
        values = values.reshape(b, l, self.embed_dim)
        outputs = self.out_proj(values)
        
        if print_opt_cs:
            return outputs, cos
        
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
    def forward(self, x, print_opt_cs=False):
        qkv = self.layernorm1(x)
        
        outputs = self.attn(qkv, print_opt_cs)
        if isinstance(outputs, tuple):
            out, cos = outputs
            x = x + out
        else:
            x = x + outputs
            
        """if print_opt_cs:
            attn, cos = self.attn(qkv, print_opt_cs)
            x = x + attn
        else:
            x = x + self.attn(qkv)"""

            
        x = x + self.linear(self.layernorm2(x))
        
        if print_opt_cs:
            return x, cos
        
        return x

    
"""
Extracted from LeYOLO 
"""
    

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v     

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p    

def adjust_channels(channels: int, width_mult: float):
    return _make_divisible(channels * width_mult, 8)


def activation_function(act="RE"):
    res = nn.Hardswish()
    if act == "RE":
        res = nn.ReLU6(inplace=True)
    elif act == "GE":
        res = nn.GELU()
    elif act == "SI":
        res = nn.SiLU()
    elif act == "EL":
        res = nn.ELU()
    else:
        res = nn.Hardswish()
    return res


class mn_conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act="RE", p=None, g=1, d=1):
        super().__init__()
        padding = 0 if k==s else autopad(k,p,d)
        self.c = nn.Conv2d(c1, c2, k, s, padding, groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = activation_function(act)#nn.ReLU6(inplace=True) if act=="RE" else nn.Hardswish()
    
    def forward(self, x):
        return self.act(self.bn(self.c(x)))
    
    
class InvertedBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, sa="None", act="RE", stride=1, pw=True):
        #input_channels, output_channels, repetition, stride, expension ratio
        super().__init__()
        #act = nn.ReLU6(inplace=True) if NL=="RE" else nn.Hardswish()
        c_mid = e if e != None else c1
        self.residual = c1 == c2 and stride == 1

        features = [mn_conv(c1, c_mid, act=act)] if pw else [] #if c_mid != c1 else []
        features.extend([mn_conv(c_mid, c_mid, k, stride, g=c_mid, act=act),
                         #attn,
                         nn.Conv2d(c_mid, c2, 1),
                         nn.BatchNorm2d(c2),
                         #nn.SiLU(),
                         ])
        self.layers = nn.Sequential(*features)
    def forward(self, x):
        #print(x.shape)
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


        



##EfficientMOd

class PatchEmbed(nn.Module):
    def __init__(self, c_in=3, embed_dim=96, patch_size=4, patch_stride=4, pad=0, norm_layer=None):
        super().__init__()

        self.proj = nn.Conv2d(c_in, embed_dim, patch_size, patch_stride, padding=pad)
        self.norm = None
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
    def forward(self, x):
        #in orginal paper , the input is b h w c which I don't get it why, since patchembed is literaly the first pass ?
        #it seems like it is because we use PatchEmbed call when downsampling.
#        x = self.proj(x).permute(0,3,1,2)
        x = self.proj(x.permute(0,3,1,2)).permute(0,2,3,1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class classic_mlp(nn.Module):
    def __init__(self, c_in, hidden_dim=None, c_out=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        c_out = c_out or c_in
        hidden_dim = hidden_dim or c_in

        self.norm1 = nn.LayerNorm(c_in)
        self.fc1 = nn.Linear(c_in, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, c_out)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   


class eff_mod_attn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = max(dim // num_heads, 32)
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, self.num_heads * self.head_dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, s, d = x.shape

        qkv = self.qkv(x).reshape(b,s,3,self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)

        x = x.transpose(1,2).reshape(b,s,self.head_dim * self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class eff_mod_attn_block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., num_heads=8, qkv_bias=False, qk_norm=False, drop=0., attn_drop=0., init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = eff_mod_attn(dim, num_heads, qkv_bias, qk_norm, drop, attn_drop, norm_layer)
        
        self.layer_scale_1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = classic_mlp(dim, int(dim*mlp_ratio), drop=drop)

        self.layer_scale_2 = LayerScale(dim, init_values) if init_values else nn.Identity()
    def forward(self, x):
        b,h,w,c = x.size()
        x = x.reshape(b,h*w,c)
        x = x + self.layer_scale_1(self.attn(self.norm1(x)))
        x = x + self.layer_scale_2(self.mlp(self.norm2(x)))
        x = x.reshape(b,h,w,c)
        return x



class ContextLayer(nn.Module):
    def __init__(self, dim, conv_dim, context_size=[3], context_act=nn.GELU, context_f=True, context_g=True):
        super().__init__()
        
        self.f = nn.Linear(dim, conv_dim) if context_f else nn.Identity()
        self.g = nn.Linear(conv_dim, dim) if context_g else nn.Identity()
        self.context_size = context_size
        self.act = context_act() if context_act else nn.Identity()

        if not isinstance(context_size, (list, tuple)):
            context_size = [context_size]
        self.context_list = nn.ModuleList()

        for c_size in context_size:
            self.context_list.append(
                nn.Conv2d(conv_dim, conv_dim, c_size, stride=1, padding=c_size//2, groups=conv_dim)
            )

    def forward(self, x):
        x = self.f(x).permute(0,3,1,2)
        out = 0
        for i in range(len(self.context_list)):
            ctx = self.act(self.context_list[i](x))
            out = out + ctx
        out = self.g(out.permute(0,2,3,1))
        return out

class mlp_ctx(nn.Module):
    def __init__(self, c_in, hidden_dim=None, c_out=None, act_layer=nn.GELU, drop=0., bias=True, conv_group_dim=4, context_size=3, context_act=nn.GELU, context_f=True, context_g=True):
        super().__init__()
        c_out = c_out or c_in
        hidden_dim = hidden_dim or c_in

        self.linear_1 = nn.Linear(c_in, hidden_dim, bias=bias)
        self.conv_group_dim = conv_group_dim
        conv_dim = hidden_dim // conv_group_dim
        self.ctx = ContextLayer(c_in, conv_dim, context_size, context_act, context_f, context_g)

        self.linear_2 = nn.Linear(hidden_dim, c_out, bias=bias)

        if hidden_dim == c_in and conv_group_dim == 1:
            self.expand_dim = False
        else:
            self.expand_dim = True
            self.act = act_layer()
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        conv_x = self.ctx(x)
        x = self.linear_1(x)
        if self.expand_dim:
            x = self.drop( self.act(x) )
            x = x * conv_x.repeat(1,1,1,self.conv_group_dim)
        else:
            x = x * conv_x

        x = self.linear_2(x)
        return x

class base_eff_mod_block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, bias=True, use_layer_scale=False, layer_scale_init_value=1e-4, conv_group_dim=4, context_size=3, context_act=nn.GELU, context_f=True, context_g=True):
        super().__init__()

        self.norm = norm_layer(dim)
        self.mlp = mlp_ctx(dim, int(dim*mlp_ratio), act_layer=act_layer,drop=drop, bias=bias, conv_group_dim=conv_group_dim, context_size=context_size, context_act=context_act, context_f=context_f, context_g=context_g)
        self.gamma_1 = 1.0
        if use_layer_scale:
            self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.gamma_1 * self.mlp(self.norm(x))
        return x

class base_eff_mod_layer(nn.Module):
    def __init__(self, dim, out_dim, depth, mlp_ratio=4., attn_ratio=4., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 bias=True, use_layer_scale=False, layer_scale_init_value=1e-4, conv_group_dim=4, context_size=3, 
                 context_act=nn.GELU,context_f=True, context_g=True, downsample=False, patch_size=3, patch_stride=2, patch_pad=1, patch_norm=True, attention_depth=0.):
        super().__init__()

        if not isinstance(mlp_ratio, (list, tuple)):
            mlp_ratio = [mlp_ratio] * depth
        if not isinstance(conv_group_dim, (list, tuple)):
            conv_group_dim = [conv_group_dim] * depth
        if not isinstance(context_size, (list, tuple)):
            context_size = [context_size] * depth

        self.blocks = nn.ModuleList([
            base_eff_mod_block(dim, mlp_ratio[i], drop, act_layer, norm_layer, bias,use_layer_scale,layer_scale_init_value,conv_group_dim[i], context_size[i], context_act, context_f, context_g)
        for i in range(depth)])
        #
        if attention_depth > 0:
            for j in range(attention_depth):
                self.blocks.append(eff_mod_attn_block(dim, attn_ratio, drop=drop, act_layer=act_layer, norm_layer=norm_layer))
        
        if downsample:
            self.downsample = PatchEmbed(dim, out_dim, patch_size, patch_stride, patch_pad, norm_layer if patch_norm else None)
        else:
            self.downsample = None

    def forward(self, x):
        for hidden_layer in self.blocks:
            x = hidden_layer(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x














###LEVIT

def levit_like_stem(c_in, c_out):
    return nn.Sequential(cbn(c_in, 32, 3, 2, 1),
                         cbn(32, 64, 3, 2, 1),   
                         cbn(64, 128, 3, 2, 1),
                         cbn(128, c_out, 3, 2, 1),)

###SWIFT FORMER

def swift_former_stem(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out//2, 3, 2, 1),
        nn.BatchNorm2d(c_out//2),
        nn.ReLU(),
        nn.Conv2d(c_out//2, c_out, 3, 2, 1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )



class ConvEncoder(nn.Module):
    def __init__(self, dim, hidden_dim=64, kernel_size=3, use_layer_scale=True):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()

        self.pwconv2 = nn.Conv2d(hidden_dim, dim, 1)
        
        self.use_layer_scale = use_layer_scale

        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.use_layer_scale:
            x = input + self.layer_scale * x
        else:
            x = input + x

        return x



class Mlp(nn.Module):
    def __init__(self, c_in, hidden_dim=None, c_out=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        c_out = c_out or c_in
        hidden_dim = hidden_dim or c_in

        self.norm1 = nn.BatchNorm2d(c_in)
        self.fc1 = nn.Conv2d(c_in, hidden_dim, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_dim, c_out, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def cos_sim_b(x_prev, x_next):
    cos_sim = torch.nn.CosineSimilarity(dim=2)
    t = cos_sim(x_prev, x_next)
    
    #t = (t - t.min()) / (t.max() - t.min()) at first I did this, but we know that maxx and minx are 1 and -1 for cosine similarity
    t = (t - (-1)) / 2 


    return t.mean()



def pw_cos_sim(x):
    #x is B H S D 
    #while in EAA it is B S D*H
    #B H S D -> B H D S -> B H*D S -> B S H*D
    #print(x.shape)
    """x = x.permute(0,1,3,2).flatten(1,2).permute(0,2,1).detach()
    t = nn.CosineSimilarity(dim=1)(x[:,:,None,:], x[:,:,:,None])
    t = (t - (-1)) / 2
    return t.mean()"""
    b,h,s,d = x.size()

    t = x.view(b,h,-1)
    t = F.normalize(t, p=2, dim=2)
    mat_sim = torch.bmm(t, t.transpose(1,2))

    #remove identity
    ide = torch.eye(h, device=t.device).unsqueeze(0).expand(b, -1, -1)

    mat_sim = mat_sim - ide

    t = mat_sim.sum() / (b*h*(h-1)) #return total similar identity
    t = (t - (-1)) / 2
    return t


def cos_sim_in(x):
    #x = x.flatten(1,2)
    t = nn.CosineSimilarity(dim=2)(x[:,None,:], x[:,:,None])
    t = (t - (-1)) / 2
    return t.mean()

#EfficientAdditiveAttention
class EAA(nn.Module):
    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.q = nn.Linear(in_dims, token_dim*num_heads)
        self.k = nn.Linear(in_dims, token_dim*num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))

        self.scale_factor = token_dim **-0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.out = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x, print_opt_cs=False):
        #print("DEBUG ! :")
        #print(x.shape)
        Q = self.q(x)
        K = self.k(x)

        Q = torch.nn.functional.normalize(Q, dim=-1)
        K = torch.nn.functional.normalize(K, dim=-1)

        Q_weight = Q @ self.w_g #BxNxD @ Dx1 = BxNx1

        A = Q_weight * self.scale_factor
        A = torch.nn.functional.normalize(A, dim=1)

        G = torch.sum(A * Q, dim=1)
        G = repeat(G, "b d -> b repeat d", repeat=K.shape[1])

        res = self.Proj(G * K) + Q

        res = self.out(res)
       
        if print_opt_cs:
            cos = cos_sim_in(res)
            return res, cos
        return res


class swift_mha_encoder(nn.Module):
    def __init__(self, dim, mlp_ratio=4., heads=4, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.local_representation = ConvEncoder(dim, dim, kernel_size=3, use_layer_scale=False)
        self.attn = MultiHeadAttention(dim, dim, heads)
        self.linear = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x,print_opt_cs=False):
        cos = None
        x = self.local_representation(x)
        b,c,h,w = x.shape
        attn = self.attn(x.permute(0,2,3,1).reshape(b,h*w,c), print_opt_cs)
        if isinstance(attn, tuple):
            attn, cos = attn

        x = x + attn.reshape(b,h,w,c).permute(0,3,1,2)
        x = x + self.linear(x) 
        if print_opt_cs:
            return x, cos
        return x


class SwiftFormerEncoder(nn.Module):

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, drop=0., use_layer_scale=False, layer_scale_init_value=1e-5):
        super().__init__()

        self.local_representation = ConvEncoder(dim, dim, kernel_size=3, use_layer_scale=False)
        self.attn = EAA(dim, dim, 1)
        self.linear = Mlp(dim, int(dim*mlp_ratio), act_layer=act_layer, drop=drop)
        
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x, print_opt_cs=False):
        x = self.local_representation(x)
        b,c,h,w = x.shape
        cos = None
        y = self.attn(x.permute(0,2,3,1).reshape(b, h*w, c), print_opt_cs)
        if isinstance(y, tuple):
            y, cos = y
        if self.use_layer_scale:
            x = x + self.layer_scale_1 * y.reshape(b,h,w,c).permute(0,3,1,2)
            x = x + self.layer_scale_2 * self.linear(x)
        else:
            x = x + y.reshape(b,h,w,c).permute(0,3,1,2)
            x = x + self.linear(x)
        
        if cos is not None:
            return x, cos
        return x




