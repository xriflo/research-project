'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.autograd import Variable
import torch.optim as optim
from random import randint
from roselabfolder import ActivityImageFolder
import torchvision.transforms as transforms

class Inception(nn.Module):
    # self.a3 = Inception(  192,        64,     96,         128,    16,         32,     32)
    def __init__(self,      in_planes,  n1x1,   n3x3red,    n3x3,   n5x5red,    n5x5,   pool_planes):

        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),
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
            nn.Dropout2d(p=0.5),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        #print ("y1:           ", y1.size())
        #print ("y2:           ", y2.size())
        #print ("y3:           ", y3.size())
        #print ("y4:           ", y4.size())
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        self.in_planes = 96
        self.n1x1 = 32
        self.n3x3red = 48
        self.n3x3 = 64
        self.n5x5red = 8
        self.n5x5 = 16
        self.pool_planes = 16
        self.num_features = 512
        self.lin_input = (self.n1x1 + self.n3x3 + self.n5x5 + self.pool_planes) * 48 * 42
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, self.in_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(True),
        )
      # self.a3 = Inception(  192,        64,     96,         128,    16,         32,     32)
        self.a3 = Inception(self.in_planes, self.n1x1, self.n3x3red, self.n3x3, self.n5x5red, self.n5x5, self.pool_planes)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.linear = nn.Linear(self.lin_input, self.num_features)

    def forward(self, x):
        out = self.pre_layers(x)
        #print(out.size())
        out = self.a3(out)
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


torch.manual_seed(1)
BATCH_SIZE = 9
SEQ_SIZE = 16
NO_CHANNELS = 3
WIDTH = 48
HEIGHT = 42
NUM_CLASSES = 5
NUM_FEATURES = 120

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.no_classes = 2
        self.num_features = 120
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(1008, self.num_features)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        #print(out.size())
        out = F.relu(self.fc1(out))
        return out


class CLSTM(nn.Module):
    def __init__(self, seq_size):
        super(CLSTM, self).__init__()
        self.batch_size = 10
        self.seq_size = 16
        self.no_channels = 3
        self.width = 48
        self.height = 42
        self.cnn = LeNet()
        self.rnn = nn.LSTM(NUM_FEATURES, NUM_CLASSES, 2)
        #self.lstm = 
    def forward(self, x):
        cnn_outputs = []
        for t in range(self.seq_size):
            #print(x.size())
            cnn_output = self.cnn(x[:,t,:,:,:])
            #print("out: ", cnn_output.size())
            cnn_outputs.append(cnn_output)
        stack_cnns = torch.stack(cnn_outputs, dim=0)
        #print("stack size: ", stack_cnns)
        output, hn = self.rnn(stack_cnns)
        return output, hn
        #for cnn_output in cnn_outputs:
best_acc = 0 

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.32950327, 0.32950327, 0.32950327), (0.2481389, 0.2481389, 0.2481389)),
])


trainset = ActivityImageFolder("../../archives_mihai/train_activity", transform_train)
#testset = ActivityImageFolder("../../archives_mihai/test_activity", transform_train)
'''
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)

use_cuda = False
net = CLSTM(SEQ_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)


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
        outputs, _ = net(inputs)
        loss = criterion(outputs.mean(dim=0), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs[-1].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        #print(outputs[-1].size(), " vs ", targets.size())

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

        outputs, _ = net(inputs)
        
        loss = criterion(outputs.mean(dim=0), targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs[-1].data, 1)
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


for epoch in range(10):
    tr_acc = train(epoch, net)
    ts_acc = test(epoch, net)
    print("train: ", tr_acc)
    print("test: ", ts_acc)
'''