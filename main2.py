import matplotlib.pyplot as plt
import torchvision

from torchvision.transforms import Compose, ToPILImage, ToTensor
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from torch.autograd import Variable



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.485, 0.485 ],
                         std = [ 0.229, 0.229, 0.229 ]),
])



train_data = datasets.ImageFolder("../datasets/raw_output_train", transform)
test_data = datasets.ImageFolder("../datasets/raw_output_test", transform)

train = torch.utils.data.DataLoader(train_data, batch_size=100)
#test_data = torch.utils.data.DataLoader(train_data, batch_size=2)


net = GoogLeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.01, weight_decay=5e-4)

net.train()
train_loss = 0

for i in range(2):
        total = 0
        correct = 0
        batch = 0
        for batch_idx, (inputs, targets) in enumerate(train):
            batch = batch + 1
            print(batch)
            #print("batch_idx ", batch_idx)
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)

            outputs = net(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            q, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            #print(torch.max(outputs.data, 1))
        print( "correct ", correct, " out of total ", total)











#data = create_posture_dataset()
#loader = torch.utils.data.DataLoader(data, batch_size=2)


'''
def get_mean_and_std(dataset, max_load=10000):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    N = min(max_load, len(dataset))
    for i in range(N):
        im,_ = dataset[i]
        for j in range(3):
            mean[j] += im[j,:,:].mean()
            std[j] += im[j,:,:].std()
    mean.div_(N)
    std.div_(N)
    return mean, std
'''

#dataiter = iter(loader)
#images, labels = dataiter.next()

#to_pil = torchvision.transforms.ToPILImage()
#c = to_pil(images[0,:,:,:])
#print(images[0,:,:,:])
#imgplot = plt.imshow(c,cmap=plt.get_cmap('gray'),)
#plt.show()

#imgplot = plt.imshow(torchvision.utils.make_grid(images))

#dataiter = iter(loader)
    #images, labels = dataiter.next()


    #images = images[:.0,:,:].reshape(images.size(0),1,images.size(2). images.size(3))
    #print(images[0,:,:,:].sum())
    #print(images[1,:,:,:].sum())
    #imgplot = plt.imshow(torchvision.utils.make_grid(images))



