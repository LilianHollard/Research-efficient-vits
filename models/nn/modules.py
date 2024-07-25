import torch
import torch.nn as nn
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
