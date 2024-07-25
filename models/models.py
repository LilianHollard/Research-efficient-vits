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
