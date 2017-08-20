'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Inception(nn.Module):
    # self.a3 = Inception(  192,        64,     96,         128,    16,         32,     32)
    def __init__(self,      in_planes,  n1x1,   n3x3red,    n3x3,   n5x5red,    n5x5,   pool_planes):

        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        #print "y1:           ", y1.size()
        #print "y2:           ", y2.size()
        #print "y3:           ", y3.size()
        #print "y4:           ", y4.size()
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.linear = nn.Linear(258048, 3)

    def forward(self, x):
        out = self.pre_layers(x)
        #print "pre_layers:   ", out.size()

        #print("------------a3-----------")
        out = self.a3(out)
        #print "after a3:     ", out.size()

        #print("---------maxpool---------")
        #out = self.maxpool(out)
        #print "out size:     ", out.size()
        
        #print("------------fc-----------")
        out = out.view(out.size(0), -1)
        #print "fc size:      ", out.size()

        #print("------------lin-----------")
        out = self.linear(out)
        #print "lin size:     ", out.size()

        #print("---------network ends--------")

        
        
        return out