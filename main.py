from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


import os
import argparse

from models import *
from utils import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.pylab as plb


parser = argparse.ArgumentParser(description='Posture estimation')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', default=10, help='no of epochs')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.32950327, 0.32950327, 0.32950327), (0.2481389, 0.2481389, 0.2481389)),
])


trainset = datasets.ImageFolder("../raw_train_resized", transform_train)
testset = datasets.ImageFolder("../raw_test_resized", transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

net = DPN()



models =    {
            "DenseNet": DenseNet121, 
            "DPN": DPN, 
            "GoogLeNet": GoogLeNet, 
            "LeNet": LeNet, 
            "MobileNet": MobileNet, 
            "ResNet": ResNet, 
            "ResNeXt": ResNeXt, 
            "ShuffleNet": ShuffleNet, 
            "VGG": VGG
            }

def train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("train: ", acc)
    return acc

def test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()


    # Save checkpoint.
    acc = 100.*correct/total
    print("test: ", acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    return acc


for name, model in models.items():
    print("Training ", name, "...")
    net = model()
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_acc = []
    test_acc = []
    for epoch in range(args.epochs):
        tr_acc = train(epoch, net)
        ts_acc = test(epoch, net)
        train_acc.append(tr_acc)
        test_acc.append(ts_acc)

    plt.plot ( plb.arange(1,args.epochs+1),train_acc,color='g',label='train acc' )
    plt.plot ( plb.arange(1,args.epochs+1),test_acc,color='r',label='test acc' )
    plt.xlabel('No epochs')
    plt.ylabel('Accuracy')
    plt.title(name + ' - training and testing')
    plt.legend(('train acc','test acc'))
    plt.savefig("results/"+name+".png")
    plt.clf()