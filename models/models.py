import torch
import torch.nn as nn
import math
from models.nn.modules import *


class custom_dino(nn.Module):
    #take dinov2 official repo torch.hub.load(...) as parameter
    def __init__(self, net):
        super().__init__()
        # self.proj = nn.Conv2d(3,384, kernel_size=(4,4), stride=(4,4), bias=False)
        # only blocks since the patch embeding canno't work with cifar 32-by-32 size.

        self.proj = net.patch_embed
        self.backbone = nn.Sequential(*net.blocks)
        # last channels from dinov2 vits14 (need a better code to extract last channel number)
        self.fc = nn.Linear(384, 10)

    def forward(self, x):
        x = self.proj(x)
        # print(x.shape)
        # x = x.flatten(2,3).transpose(2,1)
        x = self.backbone(x)
        x = torch.nn.functional.adaptive_avg_pool1d(x.transpose(2, 1), 1).squeeze(-1)
        return self.fc(x)


class ResNet(nn.Module):
    def __init__(self, num_layers, num_classes = 10, distillation=False):
        super().__init__()

        if num_layers == 18:
            layers = [2,2,2,2]
            self.expansion = 1

        self.distillation = distillation

        self.in_channels = 64
        #All ResNets (18 to 152) contain a Conv2d > BN > ReLU for the first three layers.
        self.stem = nn.Sequential(
                cba(3, self.in_channels, 7, 2, 3, bias=False),
                #nn.MaxPool2d(3,2,1)
        )
        self.layers = []
        self.layers.append(residual_layer(self.in_channels, self.in_channels, repeat=layers[0]))
        self.layers.append(residual_layer(self.in_channels, self.in_channels*2, stride=2, repeat=layers[1]))
        self.layers.append(residual_layer(self.in_channels*2, self.in_channels*4, stride=2, repeat=layers[2]))
        self.layers.append(residual_layer(self.in_channels*4, self.in_channels*8, stride=2, repeat=layers[3]))
        self.layers = nn.ModuleList(self.layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(self.in_channels*8, num_classes)

    def forward(self, x, print_opt_cs=False):
        total_cos = []
        x = self.stem(x)
        
        for hidden_layer in self.layers:
            x = hidden_layer(x, print_opt_cs)
            if isinstance(x, tuple):
                x, cos = x
                total_cos.append(cos)
        
        x = self.avgpool(x).flatten(2,3).squeeze(2)

        x = self.head(x)
        if print_opt_cs:
            return x, total_cos

        return x



class StarNet(nn.Module):
    def __init__(self, dim=32, depths=[3,3,12,5], mlp_ratio=4, drop_path_rate=0.0, num_classes=10, **kwargs):
        super().__init__()
        #depths = [3,3,8,4] #calma sur cifar10 quand même
        self.in_ch = 32 #no matter of the dim described in parameters^
        #look a lot like MobileNets and co
        self.stem = nn.Sequential(
                                    cbn(3, self.in_ch, 3, 2, 1), nn.ReLU6()
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] #stochastic depth
        
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            embed_dim = dim * 2 ** i
            #down_sampler = cbn(self.in_ch, embed_dim, 3, 1, 1) if i == 0 else cbn(self.in_ch, embed_dim, 3, 2, 1) #only for cifar10
            down_sampler = cbn(self.in_ch, embed_dim, 3, 2, 1)
            self.in_ch = embed_dim

            blocks = [star_block(self.in_ch, mlp_ratio, dpr[cur+i]) for i in range(depths[i])]
            cur += depths[i]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        #head
        self.norm = nn.BatchNorm2d(self.in_ch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_ch, num_classes) #in_ch == last output in the loop
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for hidden_layer in self.stages:
            x = hidden_layer(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)



class SquareNet(nn.Module):
    def __init__(self, dim=32, depths=[3,3,12,5], mlp_ratio=4, drop_path_rate=0.0, num_classes=10, **kwargs):
        super().__init__()
    
        self.in_ch = 32 #no matter of the dim described in parameters^
        #look a lot like MobileNets and co
        self.stem = nn.Sequential(
                                    cbn(3, self.in_ch, 3, 2, 1), nn.ReLU6()
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] #stochastic depth
        
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            embed_dim = dim * 2 ** i
            #down_sampler = cbn(self.in_ch, embed_dim, 3, 1, 1) if i == 0 else cbn(self.in_ch, embed_dim, 3, 2, 1) #only for cifar10
            down_sampler = cbn(self.in_ch, embed_dim, 3, 2, 1)
            self.in_ch = embed_dim

            blocks = [star2_t_block(self.in_ch, mlp_ratio, dpr[cur+i]) for i in range(depths[i])]
            cur += depths[i]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        #head
        self.norm = nn.BatchNorm2d(self.in_ch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_ch, num_classes) #in_ch == last output in the loop
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for hidden_layer in self.stages:
            x = hidden_layer(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)
        



class ShuffleNetV2(nn.Module):
    def __init__(self, channel_width=[24, 48, 96, 192, 1024], layer_depth=[4,8,4], num_classes=1000):
        super().__init__()

        current_idx = 0
        output_channels = channel_width[current_idx]
        self.stem = cba(3, output_channels, 3, 2, 1, bias=False)
        self.mp = nn.MaxPool2d(3, 2, 1)

        input_channels = output_channels

        layers =  []
        for repeat in layer_depth:
            current_idx += 1
            output_channels = channel_width[current_idx]

            layers.append(shufflenet_v2_block(input_channels, output_channels, 2))

            for i in range(repeat - 1):
                layers.append(shufflenet_v2_block(output_channels, output_channels, 1))
            input_channels = output_channels
        
        self.layers = nn.Sequential(*layers)
        self.out = cba(input_channels, channel_width[-1], 1, 1, 0, bias=False)

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.mp(x)
        x = self.layers(x)
        x = self.out(x)
        x = x.mean([2,3]) #global pool
        x = self.fc(x)
        return x















"""
TRANSFORMERS / HYBRID 
"""

class ViT_nocls(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads=4, num_layers=4, num_classes=10, patch_size=16, dropout=0.0, print_opt_cs=False):
        super().__init__()

        self.patch_size = patch_size
        self.print_opt_cs = print_opt_cs

        #Some are using reshape and transpose fonction to create patchenizer
        #We prefer using 2d convolutions.

        self.input_layer = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size, padding=0)

        #attn_layers = [vit_encoder(embed_dim, hidden_dim, num_heads, dropout)]
        #attn_layers.extend([vit_encoder(hidden_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])

        self.transformer = nn.Sequential(*[vit_encoder(embed_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])

        # since there is no cls tokens -> I propose to be CNN-alike: Avg pooling on spatial dim and linear on top of it.
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

        #+1 for the CLS embedding, however I am trying not to use it.
        self.pos_embedding = nn.Parameter(torch.randn(1, 1+int((224*224)/patch_size**2), embed_dim))

    def forward(self, x):
        #x <- B, C, H, W

        #1. "Patchenizer" <- using convolutions
        x = self.input_layer(x) #B, D, H, W
        x = x.flatten(2,3).permute(0, 2, 1) #B, D, N -> B, N, D
        b,n,d = x.shape

        #2. Add CLS token (I don't want to) and positional encoding (yep)
        x = x + self.pos_embedding[:,:n]#+1] #(supposed to be +1 since we need CLS token [but I don't want to :p])

        #3. Transformer
        x = self.dropout(x)
        x = self.transformer(x)#, self.print_opt_cs) #try :  print_opt_cs=False

        #x <- B, N, D
        x = x.permute(0,2,1)
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(2)
        x = self.head(x)

        return x



class hybrid_vit_cls(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads=4, num_layers=4, k=4, num_classes=10, patch_size=16, dropout=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_layers = num_layers

        self.input_layer = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size, padding=0)
        
        self.cnns = nn.Sequential(*[nn.Sequential(
                                  cba(embed_dim, int(embed_dim * 1.5), 3, 1,1),
                                  cba(int(embed_dim * 1.5), embed_dim, 3, 1,1) ) for _ in range(num_layers - k)]) #if k == num_layers, no hybridation.
        
        self.transformer = nn.ModuleList([vit_encoder(embed_dim, hidden_dim, num_heads, dropout) for _ in range(k)])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1,1 + int((32*32)/self.patch_size ** 2), self.embed_dim))

    def forward(self, x, print_cs=False):
        b = x.shape[0]
        cs = []
        self.cls_embedding = nn.Parameter(torch.ones(b, 1, self.embed_dim, device=x.device),requires_grad=True)

        x = self.input_layer(x)
        x = self.cnns(x)

        x = x.flatten(2,3).permute(0,2,1)

        x = torch.cat([self.cls_embedding, x], dim=1)
        x = x + self.pos_embedding

        x = self.dropout(x)

        total_cos = []
        for t in self.transformer:
            outputs = t(x, print_cs)
            if isinstance(outputs, tuple):
                x, cos = outputs
                total_cos.append(cos)
            else:
                x = outputs

        x = self.head(x[:,0,:]) #why tho ? Need double check plz ! 
        if print_cs:
            return x, total_cos

        return x


class SwiftFormer_full(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        c = [48, 56, 112, 220]
        r = [3, 3, 6, 4]
        s = [1,2,2]

        self.stem = swift_former_stem(3, c[0])
        
        self.all_swifts = []
        self.all_cbns = []
        for i in range(len(c)):
            #pb en faisant comme ça, le modulist n'envoi pas sur le GPU
            self.all_swifts.append(nn.ModuleList([SwiftFormerEncoder(c[i]) for _ in range(r[i])]))
            if i < len(c)-1:
                self.all_cbns.append(cbn(c[i], c[i+1], 3, s[i], 1))    


        self.all_cbns = nn.ModuleList(self.all_cbns)
        self.all_swifts = nn.ModuleList(self.all_swifts)
        
        self.bn = nn.BatchNorm2d(c[3])
        self.head = nn.Sequential(
            #nn.BatchNorm2d(c[3]),
            nn.Linear(c[3], num_classes)
        )
    def forward(self, x, print_opt_cs=False):
        total_cos = []
        if print_opt_cs:
            x = self.stem(x)
            for i in range(len(self.all_swifts)):
                for hidden_layer in self.all_swifts[i]:
                    x, cos = hidden_layer(x, print_opt_cs)
                    total_cos.append(cos)
                if i < len(self.all_swifts)-1:
                    x = self.all_cbns[i](x)
        else:
            x = self.stem(x)
            for i in range(len(self.all_swifts)):
                for hidden_layer in self.all_swifts[i]:
                    x = hidden_layer(x)
                if i < len(self.all_swifts)-1:
                    x = self.all_cbns[i](x)
            

        x = self.bn(x).flatten(2).mean(-1)
        #print(x.shape)
        x = self.head(x)
        if print_opt_cs:
            return x, total_cos
        return x


class SwiftFormer(nn.Module):
    def __init__(self, num_classes=10, channel_width = 1.0, layer_depth=1.0):
        super().__init__()
        c = [32, 64, 128, 256]
        r = [3, 3, 6, 4]
        s = [1,2,2]
        
        c = [math.ceil(x * channel_width) for x in c]
        r = [math.ceil(x * layer_depth) for x in c]

        self.stem = swift_former_stem(3, c[0])
        
        self.all_conv_encoders = []
        self.all_cbns = []
        self.all_swifts = []
        for i in range(len(c)):
            self.all_conv_encoders.append(nn.ModuleList([ConvEncoder(c[i], c[i]*4, 3) for _ in range(r[i]-1)]))#, SwiftFormer(c[i])))
            self.all_swifts.append(SwiftFormerEncoder(c[i]))
            if i < len(c)-1:
                self.all_cbns.append(cbn(c[i], c[i+1], 3, s[i], 1))    


        self.all_cbns = nn.ModuleList(self.all_cbns)
        self.all_swifts = nn.ModuleList(self.all_swifts)
        self.all_conv_encoders = nn.ModuleList(self.all_conv_encoders)
        
        self.bn = nn.BatchNorm2d(c[3])
        self.head = nn.Sequential(
            #nn.BatchNorm2d(c[3]),
            nn.Linear(c[3], num_classes)
        )
    def forward(self, x, print_opt_cs=False):
        total_cos = []
        if print_opt_cs:
            x = self.stem(x)
            for i in range(len(self.all_conv_encoders)):
                for hidden_layer in self.all_conv_encoders[i]:
                    x = hidden_layer(x)
                x, cos = self.all_swifts[i](x, print_opt_cs)
                total_cos.append(cos)
                if i < len(self.all_swifts)-1:
                    x = self.all_cbns[i](x)
        else:
            x = self.stem(x)
            for i in range(len(self.all_conv_encoders)):
                for hidden_layer in self.all_conv_encoders[i]:
                    x = hidden_layer(x)
                x = self.all_swifts[i](x)
                if i < len(self.all_swifts)-1:
                    x = self.all_cbns[i](x)
            

        x = self.bn(x).flatten(2).mean(-1)
        #print(x.shape)
        x = self.head(x)
        if print_opt_cs:
            return x, total_cos
        return x

class mini_vit_former(nn.Module):
    def __init__(self, num_classes=10, channel_width=1.0, layer_depth=1.0):
        super().__init__()
        c = [32, 64, 128, 256]
        heads = [4,4,4,4]
        r = [3, 2, 6, 4]
        s = [2,2,2]

        c = [math.ceil(x*channel_width) for x in c]
        r = [math.ceil(x*layer_depth) for x in r]

        self.stem = swift_former_stem(3, c[0])
        
        self.all_conv_encoders = []
        self.all_cbns = []
        self.all_swifts = []
        for i in range(len(c)):
            #pb en faisant comme ça, le modulist n'envoi pas sur le GPU
            self.all_conv_encoders.append(nn.ModuleList([ConvEncoder(c[i], c[i]*4, 3) for _ in range(r[i]-1)]))#, SwiftFormer(c[i])))
            self.all_swifts.append(swift_mha_encoder(c[i]))
            if i < len(c)-1:
                self.all_cbns.append(cbn(c[i], c[i+1], 3, s[i], 1))    



        self.all_cbns = nn.ModuleList(self.all_cbns)
        self.all_swifts = nn.ModuleList(self.all_swifts)
        self.all_conv_encoders = nn.ModuleList(self.all_conv_encoders)
        
        self.bn = nn.BatchNorm2d(c[3])
        self.head = nn.Sequential(
            #nn.BatchNorm2d(c[3]),
            nn.Linear(c[3], num_classes)
        )
    def forward(self, x, print_opt_cs=False):
        total_cos = []
        if print_opt_cs:
            x = self.stem(x)
            for i in range(len(self.all_conv_encoders)):
                for hidden_layer in self.all_conv_encoders[i]:
                    x = hidden_layer(x)
                x, cos = self.all_swifts[i](x, print_opt_cs)
                total_cos.append(cos)
                if i < len(self.all_swifts)-1:
                    x = self.all_cbns[i](x)
        else:
            x = self.stem(x)
            for i in range(len(self.all_conv_encoders)):
                for hidden_layer in self.all_conv_encoders[i]:
                    x = hidden_layer(x)
                x = self.all_swifts[i](x)
                if i < len(self.all_swifts)-1:
                    x = self.all_cbns[i](x)
            

        x = self.bn(x).flatten(2).mean(-1)
        #print(x.shape)
        x = self.head(x)
        if print_opt_cs:
            return x, total_cos
        return x


class le_vit_former(nn.Module):
    def __init__(self, num_classes=10, channel_width=1.0, layer_depth=1.0):
        super().__init__()
        c = [256, 384, 512]
        r = [3, 6, 4]
        s = [2,2,1]


        c = [math.ceil(x*channel_width) for x in c]
        r = [math.ceil(x*layer_depth) for x in r]

        #self.stem = swift_former_stem(3, c[0])
        self.stem = levit_like_stem(3,c[0])
        
        self.all_conv_encoders = []
        self.all_cbns = []
        self.all_swifts = []
        for i in range(len(c)):
            #pb en faisant comme ça, le modulist n'envoi pas sur le GPU
            self.all_conv_encoders.append(nn.ModuleList([ConvEncoder(c[i], c[i]*4, 3) for _ in range(r[i]-1)]))#, SwiftFormer(c[i])))
            self.all_swifts.append(swift_mha_encoder(c[i]))
            if i < len(c)-1:
                self.all_cbns.append(cbn(c[i], c[i+1], 3, s[i], 1))    



        self.all_cbns = nn.ModuleList(self.all_cbns)
        self.all_swifts = nn.ModuleList(self.all_swifts)
        self.all_conv_encoders = nn.ModuleList(self.all_conv_encoders)
        
        self.bn = nn.BatchNorm2d(c[-1])
        self.head = nn.Sequential(
            #nn.BatchNorm2d(c[3]),
            nn.Linear(c[-1], num_classes)
        )
    def forward(self, x, print_opt_cs=False):
        total_cos = []
        if print_opt_cs:
            x = self.stem(x)
            for i in range(len(self.all_conv_encoders)):
                for hidden_layer in self.all_conv_encoders[i]:
                    x = hidden_layer(x)
                x, cos = self.all_swifts[i](x, print_opt_cs)
                total_cos.append(cos)
                if i < len(self.all_swifts)-1:
                    x = self.all_cbns[i](x)
        else:
            x = self.stem(x)
            for i in range(len(self.all_conv_encoders)):
                for hidden_layer in self.all_conv_encoders[i]:
                    x = hidden_layer(x)
                x = self.all_swifts[i](x)
                if i < len(self.all_swifts)-1:
                    x = self.all_cbns[i](x)
            

        x = self.bn(x).flatten(2).mean(-1)
        #print(x.shape)
        x = self.head(x)
        if print_opt_cs:
            return x, total_cos
        return x



class mini_mlp_encoder(nn.Module):
    def __init__(self, width_scale = 1.0, length_scale=1.0, c=[32,64,128,256], r=[3,3,6,4], num_classes=10):
        super().__init__()
        c = [int(x * width_scale) for x in c]
        r = [int(x *length_scale) for x in r]

        self.stem = swift_former_stem(3, c[0])
        self.c1 = nn.Sequential(*[ConvEncoder(c[0], c[0]*4, 3) for _ in range(r[0])], cbn(c[0], c[1],3, 1, 1))
        self.c2 = nn.Sequential(*[ConvEncoder(c[1], c[1]*4, 3) for _ in range(r[1])], cbn(c[1], c[2], 3, 2, 1))
        self.c3 = nn.Sequential(*[ConvEncoder(c[2], c[2]*4, 3) for _ in range(r[2])], cbn(c[2], c[3],3, 1, 1))
        self.c4 = nn.Sequential(*[ConvEncoder(c[3], c[3]*4, 3) for _ in range(r[3])])

        self.bn = nn.BatchNorm2d(c[3])
        self.head = nn.Linear(c[3], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)

        x = self.bn(x).flatten(2).mean(-1)
        x = self.head(x)
        return x

class mini_vit(nn.Module):
    def __init__(self, width_scale = 1.0, length_scale= 1.0, c=[32,64,128,256], r=[3,3,6,4], num_classes=10):
        super().__init__()
        c = [int(x * width_scale) for x in c]
        r = [int(x *length_scale) for x in r]

        self.stem = swift_former_stem(3, c[0])

        self.c1 = nn.Sequential(*[vit_encoder(c[0], c[0]*4, 4, 0.0) for _ in range(r[0])])
        self.mid_1 = cbn(c[0], c[1], 3, 1, 1)
        self.c2 = nn.Sequential(*[vit_encoder(c[1], c[1]*4, 4, 0.0) for _ in range(r[1])])
        self.mid_2 = cbn(c[1], c[2], 3, 2, 1)
        self.c3 = nn.Sequential(*[vit_encoder(c[2], c[2]*4, 4, 0.0) for _ in range(r[2])])
        self.mid_3 = cbn(c[2], c[3], 3, 2, 1)
        self.c4 = nn.Sequential(*[vit_encoder(c[3], c[3]*4, 4, 0.0) for _ in range(r[3])])
            
        self.bn = nn.BatchNorm2d(c[3])
        self.head = nn.Linear(c[3], num_classes)

    def forward(self, x):
        #didn't implemented flatten / reshape operation inside vit_encoder (because a vit encoder doesn't need so)
        #therefore, manipulation must be done here.
        x = self.stem(x)
        b,c,h,w = x.shape
        x = self.mid_1(
                self.c1(x.permute(0,2,3,1).reshape(b,h*w, c)).reshape(b,h,w,c).permute(0,3,1,2)
        )
        b,c,h,w = x.shape
        x = self.mid_2(
                self.c2(x.permute(0,2,3,1).reshape(b,h*w, c)).reshape(b,h,w,c).permute(0,3,1,2)
        )
        b,c,h,w = x.shape
        x = self.mid_3(
                self.c3(x.permute(0,2,3,1).reshape(b,h*w, c)).reshape(b,h,w,c).permute(0,3,1,2)
        )
        b,c,h,w = x.shape
        x = self.c4(x.permute(0,2,3,1).reshape(b,h*w, c)).reshape(b,h,w,c).permute(0,3,1,2)
        x = self.bn(x).flatten(2).mean(-1)
        x = self.head(x)
        return x
        


class mini_former(nn.Module):
    def __init__(self, width_scale = 1.0, length_scale= 1.0, c=[32,64,128,256], r=[3,3,6,4], num_classes=10):
        super().__init__()
        c = [int(x * width_scale) for x in c]
        r = [int(x *length_scale) for x in r]

        self.stem = swift_former_stem(3, c[0])

        self.c1 = nn.Sequential(*[ConvEncoder(c[0], c[0]*4, 3) for _ in range(r[0])], cbn(c[0], c[1],3, 1, 1))
        self.c2 = nn.Sequential(*[ConvEncoder(c[1], c[1]*4, 3) for _ in range(r[1])], cbn(c[1], c[2], 3, 2, 1))
        self.c3 = nn.Sequential(*[ConvEncoder(c[2], c[2]*4, 3) for _ in range(r[2])], cbn(c[2], c[3], 3, 2, 1))
        #self.c3 = nn.Sequential(*[vit_encoder(c[2], c[2]*4, 4, 0.0) for _ in range(r[2])])#,cbn(c[2], c[3], 3, 2, 1))
        #self.c3_cbn = cbn(c[2], c[3], 3, 2, 1)

        self.c4 = nn.ModuleList([vit_encoder(c[3], c[3]*4, 4, 0.0) for _ in range(r[3])])
        
        self.bn = nn.BatchNorm2d(c[3])
        self.head = nn.Linear(c[3], num_classes)

    def forward(self, x, print_opt_cs=False):
        total_cos = []
        x = self.stem(x)
        x = self.c1(x)
        x = self.c2(x)
        
        #b,c,h,w = x.shape
        #x = x.permute(0,2,3,1).reshape(b, h*w, c)
        x = self.c3(x)#.reshape(b,h,w,c).permute(0,3,1,2)
        #x = self.c3_cbn(x)
        b,c,h,w = x.shape
        x = x.permute(0,2,3,1).reshape(b, h*w, c)
        for hidden_layer in self.c4:
            x = hidden_layer(x, print_opt_cs)
            if isinstance(x, tuple):
                x, cos = x
                total_cos.append(cos)
        #        x = self.c4(x.permute(0,2,3,1).reshape(b,h*w,c)).reshape(b,h,w,c).permute(0,3,1,2)
        x = x.reshape(b,h,w,c).permute(0,3,1,2)
        x = self.bn(x).flatten(2).mean(-1)
        x = self.head(x)
        if print_opt_cs:
            return x, total_cos
        return x
        

class ViT_cls(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads=4, num_layers=4, num_classes=10, patch_size=16, dropout=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
    

        #Some are using reshape and transpose fonction to create patchenizer
        #We prefer using 2d convolutions.

        self.input_layer = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size, padding=0)

        #attn_layers = [vit_encoder(embed_dim, hidden_dim, num_heads, dropout)]
        #attn_layers.extend([vit_encoder(hidden_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        
        self.transformer = nn.ModuleList([vit_encoder(embed_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        
        # since there is no cls tokens -> I propose to be CNN-alike: Avg pooling on spatial dim and linear on top of it.
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + int((32 * 32) / self.patch_size ** 2), self.embed_dim))
        
    def forward(self, x, print_cs=False):
        # x <- B, C, H, W
        b = x.shape[0]
        cs = []
        # +1 for the CLS embedding
        self.cls_embedding = nn.Parameter(torch.ones(b, 1, self.embed_dim, device=x.device),
                                          # [batch_size, number_of_tokens, embedding_dimension]
                                          requires_grad=True)  # make sure the embedding is learnable




        #1. "Patchenizer" <- using convolutions
        x = self.input_layer(x) #B, D, H, W
        x = x.flatten(2,3).permute(0, 2, 1) #B, D, N -> B, N, D

        #2. Add CLS token and positional encoding
        x = torch.cat([self.cls_embedding, x], dim=1)
        x = x + self.pos_embedding#[:,:n]#+1] already + 1 on init

        #3. Transformer
        x = self.dropout(x)
        
        total_cos = []
        for t in self.transformer:
            outputs = t(x, print_cs)
            if isinstance(outputs, tuple):
                x, cos = outputs
                total_cos.append(cos)
            else:
                x = outputs
         
        #head only on cls embedding it seems
        x = self.head(x[:,0,:])
        if print_cs:
            return x, total_cos
        
        return x
    
    
"""
LeYOLO
"""


class leyolo_backbone(nn.Module):
    def __init__(self, k = 16, ch = [1,2,4,8], s = [1,2,2,2], e = [3, 3, 3, 3]):
        super().__init__()
        #def __init__(self, c1, c2, k=3, e=None, sa="None", act="RE", stride=1, pw=True):
        #InvertedBottleneck()
        #mn_conv first, then inverted bottleneck - at least 6 ?
        
        #class mn_conv(nn.Module):
        #def __init__(self, c1, c2, k=1, s=1, act="RE", p=None, g=1, d=1):
        
        self.stem = nn.Sequential(mn_conv(3, k, 3, 1),  #16x16
                                  mn_conv(k, k, 1),
                                 )
        c = k
        layers = []
        for i in range(len(ch)):
            layers.append(nn.Sequential(InvertedBottleneck(c, k*ch[i], 3, k*e[i], "None", "SI", s[i]),
                                        InvertedBottleneck(k*ch[i], k*ch[i], 3, k*ch[i]*3, "None", "SI", 1)
                                       )
                         )
            c = k*ch[i]
        
        self.layers = nn.Sequential(*layers)
        
        self.fc = nn.Linear(c*4*4, 10)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x=x.flatten(2,3).flatten(1,2)
        x = self.fc(x)
        return x
        








##EFFICIENT Mod



class EfficientMod(nn.Module):
    def __init__(self, c_in=3, num_classes=10, patch_size=[4,3,3,3], patch_stride=[4,2,2,2], patch_pad=[0,1,1,1], patch_norm=True, 
                 embed_dim=[64, 128, 256, 512], depths=[2,2,6,2], attention_depth=[0,0,0,0], mlp_ratio=[4.0,4.0,4.0,4.0], attn_ratio=[4.0,4.0,4.0,4.0], act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_layer_scale=False, layer_scale_value=1e-4,bias=True, drop=0., conv_group_dim=[4,4,4,4], context_size=[3,3,3,3], context_act=nn.GELU, context_f=True, context_g=True, **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.stem = PatchEmbed(c_in, embed_dim[0], patch_size[0], patch_stride[0], patch_pad[0], norm_layer)
        
        #might add stochastic depth
        
        """def __init__(self, dim, out_dim, depth, mlp_ratio=4., attn_ratio=4., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 bias=True, use_layer_scale=False, layer_scale_init_value=1e-4, conv_group_dim=4, context_size=3, 
                 context_act=nn.GELU,context_f=True, context_g=True, downsample=None, patch_size=3, patch_stride=2, patch_pad=1, patch_norm=True, attention_depth=0.):"""

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                base_eff_mod_layer(embed_dim[i], embed_dim[i + 1] if (i < self.num_layers -1) else None , depths[i], mlp_ratio[i], attn_ratio[i], drop, act_layer, norm_layer, 
                           bias, use_layer_scale, layer_scale_value, conv_group_dim[i], context_size[i], context_act, context_f, context_g,
                           (i < self.num_layers - 1), 
                           patch_size[i+1] if (i < self.num_layers -1) else None,
                           patch_stride[i+1] if (i < self.num_layers -1) else None,
                           patch_pad[i+1] if (i < self.num_layers -1) else None,
                           patch_norm,
                           attention_depth=attention_depth[i]
                            )
            )

        self.norm = norm_layer(embed_dim[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dim[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x.permute(0,2,3,1))
        for hidden_layer in self.layers:
            x = hidden_layer(x)
        x = self.norm(x)
        x = self.avgpool(x.permute(0,3,1,2))
        x = torch.flatten(x,1)
        
        x = self.head(x)
        return x


def efficientMod_xxs(pretrained=False, **kwargs):
    depths = [2, 2, 6, 2]
    attention_depth = [0, 0, 1, 2]
    att_ratio = [0, 0, 4, 4]
    mlp_ratio = [
        [1, 6, 1, 6],
        [1, 6, 1, 6],
        [1, 6] * 3,
        [1, 6, 1, 6],
    ]
    context_size = [
        [7] * 10,
        [7] * 10,
        [7] * 20,
        [7] * 10,
    ]
    conv_group_dim = mlp_ratio
    model = EfficientMod(c_in=3, num_classes=10,
                      patch_size=[7, 3, 3, 3], patch_stride=[4, 2, 2, 2], patch_pad=[3, 1, 1, 1], patch_norm=True,
                      embed_dim=[32, 64, 128, 256], depths=depths, attention_depth=attention_depth,
                      mlp_ratio=mlp_ratio, att_ratio=att_ratio,
                      act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layerscale=True, layerscale_value=1e-4,
                      bias=True, drop_rate=0.,
                      conv_group_dim=conv_group_dim, context_size=context_size, context_act=nn.GELU,
                      context_f=True, context_g=True,
                      )
    return model
