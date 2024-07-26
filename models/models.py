import fontTools.merge
import torch
import torch.nn as nn

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
                nn.MaxPool2d(3,2,1)
        )
        self.layer1 = residual_layer(self.in_channels, self.in_channels, repeat=layers[0])
        self.layer2 = residual_layer(self.in_channels, self.in_channels*2, stride=2, repeat=layers[1])
        self.layer3 = residual_layer(self.in_channels*2, self.in_channels*4, stride=2, repeat=layers[2])
        self.layer4 = residual_layer(self.in_channels*4, self.in_channels*8, stride=2, repeat=layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(self.in_channels*8, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))

        x = self.avgpool(x).flatten(2,3).squeeze(2)

        x = self.head(x)
        return x

class ViT_nocls(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads=4, num_layers=4, num_classes=10, patch_size=16, dropout=0.0):
        super().__init__()

        self.patch_size = patch_size

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
        x = self.transformer(x)

        #x <- B, N, D
        x = x.permute(0,2,1)
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(2)
        x = self.head(x)

        return x



class ViT_cls(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads=4, num_layers=4, num_classes=10, patch_size=16, dropout=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size

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
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + int((224 * 224) / self.patch_size ** 2), self.embed_dim))
    def forward(self, x):
        # x <- B, C, H, W
        b = x.shape[0]

        # +1 for the CLS embedding
        self.cls_embedding = nn.Parameter(torch.ones(b, 1, self.embed_dim),
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
        x = self.transformer(x)

        #head only on cls embedding it seems
        x = self.head(x[:,0,:])

        return x