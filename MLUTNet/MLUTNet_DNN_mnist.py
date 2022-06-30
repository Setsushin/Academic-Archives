from __future__ import print_function

import argparse
import os
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import dataset
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix
from plotcm import plot_confusion_matrix
from modules import (Binarize, BinarizeConv2d, BinarizeLinear, HingeLoss)

# Training settings
parser = argparse.ArgumentParser(description='MLUTNet v0.9.0')
parser.add_argument('--batch-size', type=int, default=256, metavar='Num',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='Num',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--LR-mode', type=str, default='CosineAnnWarm',
                    help='{Decay, CosineAnn, CosineAnnWarm}')
parser.add_argument('--exp-number', type=int, default=1,
                    help='1:DNN, 2:BNN, 3:MLUTNET, 4:B-MLUTNET')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='dataset choice (mnist, kmnist, fashionmnist)')

parser.add_argument('--input-size', type=int, default=784,
                    help='input layer size')
parser.add_argument('--hidden-size', type=int, default=256 *3,
                    help='hidden layer size')
parser.add_argument('--output-size', type=int, default=10,
                    help='output layer size')

# 从检查点恢复
parser.add_argument('--resume', action='store_true', default=False,
                    help='whether resume from checkpoint')
parser.add_argument('--checkpointspath', type=str, default='/checkpoints/mnist-MLP-RMSprop-CosineAnnWarm-epochs40.pth',
                    help='dataset choice (mnist, kmnist, fashionmnist)')
# 不使用cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# 随机数种子
parser.add_argument('--seed', type=int, default=519, metavar='S',
                    help='random seed (default: 1)')
# step记录间隔
parser.add_argument('--log-interval', type=int, default=10, metavar='L',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available() # 参数里使用cuda并且cuda可用才使用cuda

# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

class DNN(nn.Module):

    def __init__(self, D_in, H, D_out):
        super(DNN, self).__init__()
        self.__name__ = 'DNN'
        # self.infl_ratio=3
        # self.neuron_of_hiddenlayers = 100
        self.d_in = D_in
        self.d_out = D_out
        self.H = H

        self.logsoftmax=nn.LogSoftmax(dim=1)
        self.drop = nn.Dropout(0.5)

        self.input = nn.Linear(D_in, H)
        self.bn1 = nn.BatchNorm1d(H)
        self.act1 = nn.ReLU()
        self.middle1 = nn.Linear(H, H)
        self.bn2 = nn.BatchNorm1d(H)
        self.act2 = nn.ReLU()
        self.middle2 = nn.Linear(H, H)
        self.bn3 = nn.BatchNorm1d(H)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(H, D_out)

    def forward(self, x):
        x = x.view(-1, self.d_in)
        x = self.input(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.middle1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.middle2(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.output(x)
        return self.logsoftmax(x)

class BNN(nn.Module):

    def __init__(self, D_in, H, D_out):
        super(BNN, self).__init__()
        self.__name__ = 'BNN'
        self.d_in = D_in

        self.logsoftmax=nn.LogSoftmax(dim=1)
        self.drop = nn.Dropout(0.5)

        self.input = BinarizeLinear(D_in, H)
        self.act1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(H)
        self.middle1 = BinarizeLinear(H, H)
        self.act2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(H)
        self.middle2 = BinarizeLinear(H, H)
        self.act3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(H)
        self.output = BinarizeLinear(H, D_out)

    def forward(self, x):
        x = x.view(-1, self.d_in)
        x = self.input(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.middle1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.middle2(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.output(x)
        return self.logsoftmax(x)

class MLUTNet(nn.Module):

    def __init__(self, D_in, H, D_out):
        super(MLUTNet, self).__init__()
        self.__name__ = 'MLUTNet'
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.drop = nn.Dropout(0.5)

        self.d_in = D_in

        self.sub_nn_block_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("act1", nn.ReLU())
                ]
            ))
        
        self.sub_nn_block_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("act1", nn.ReLU())
                ]
            ))

        self.sub_nn_block_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("act1", nn.ReLU())
                ]
            ))
        
        self.sub_nn_block_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("act1", nn.ReLU())
                ]
            ))

        self.sub_nn_block_5 = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(H, D_out))
                ]
            ))

        self.sub_nn_block_6 = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(H, D_out))
                ]
            ))

    def forward(self, x):
        x = x.view(-1, self.d_in * 2) 
        i1, i2 = torch.split(x, [self.d_in, self.d_in], dim = 1)
        a1 = self.sub_nn_block_1(i1)
        a2 = self.sub_nn_block_2(i2)
        a3 = self.sub_nn_block_3(a1)
        a4 = self.sub_nn_block_4(a1 + a2)
        a5 = self.sub_nn_block_5(a3 + a4)
        a6 = self.sub_nn_block_6(a4)
        y = torch.cat((a5, a6), dim = 1)
        return self.logsoftmax(y)

class B_MLUTNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(B_MLUTNet, self).__init__()
        self.__name__ = 'B_MLUTNet'
        self.logsoftmax = nn.LogSoftmax(dim=1) # remember add dim=X when define
        self.drop = nn.Dropout(0.5)
        self.d_in = D_in

        self.sub_nn_block_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))

        self.sub_nn_block_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))

        self.sub_nn_block_5 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, D_out))
                ]
            ))

        self.sub_nn_block_6 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, D_out))
                ]
            ))

    def forward(self, x):
        x = x.view(-1, self.d_in * 2) 
        i1, i2 = torch.split(x, [self.d_in, self.d_in], dim = 1)
        a1 = self.sub_nn_block_1(i1)
        a2 = self.sub_nn_block_2(i2)
        a3 = self.sub_nn_block_3(a1)
        a4 = self.sub_nn_block_4(a1 + a2)
        a5 = self.sub_nn_block_5(a3 + a4)
        a6 = self.sub_nn_block_6(a4)
        y = torch.cat((a5, a6), dim = 1)
        return self.logsoftmax(y)

class MLUTNet_fc(nn.Module):

    def __init__(self, D_in, H, D_out):
        super(MLUTNet_fc, self).__init__()
        self.__name__ = 'MLUTNet_fc'
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.drop = nn.Dropout(0.5)

        self.d_in = D_in

        self.sub_nn_block_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("act1", nn.ReLU())
                ]
            ))
        
        self.sub_nn_block_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("act1", nn.ReLU())
                ]
            ))

        self.sub_nn_block_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("act1", nn.ReLU())
                ]
            ))
        
        self.sub_nn_block_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("act1", nn.ReLU())
                ]
            ))

        self.sub_nn_block_5 = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(H, D_out))
                ]
            ))

        self.sub_nn_block_6 = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(H, D_out))
                ]
            ))

    def forward(self, x):
        x = x.view(-1, self.d_in * 2) 
        i1, i2 = torch.split(x, [self.d_in, self.d_in], dim = 1)
        a1 = self.sub_nn_block_1(i1)
        a2 = self.sub_nn_block_2(i2)
        a3 = self.sub_nn_block_3(a1 + a2)
        a4 = self.sub_nn_block_4(a1 + a2)
        a5 = self.sub_nn_block_5(a3 + a4)
        a6 = self.sub_nn_block_6(a3 + a4)
        y = torch.cat((a5, a6), dim = 1)
        return self.logsoftmax(y)

class B_MLUTNet_fc(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(B_MLUTNet_fc, self).__init__()
        self.__name__ = 'B_MLUTNet_fc'
        self.logsoftmax = nn.LogSoftmax(dim=1) # remember add dim=X when define
        self.drop = nn.Dropout(0.5)
        self.d_in = D_in

        self.sub_nn_block_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))

        self.sub_nn_block_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))

        self.sub_nn_block_5 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, D_out))
                ]
            ))

        self.sub_nn_block_6 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, D_out))
                ]
            ))

    def forward(self, x):
        x = x.view(-1, self.d_in * 2) 
        i1, i2 = torch.split(x, [self.d_in, self.d_in], dim = 1)
        a1 = self.sub_nn_block_1(i1)
        a2 = self.sub_nn_block_2(i2)
        a3 = self.sub_nn_block_3(a1 + a2)
        a4 = self.sub_nn_block_4(a1 + a2)
        a5 = self.sub_nn_block_5(a3 + a4)
        a6 = self.sub_nn_block_6(a3 + a4)
        y = torch.cat((a5, a6), dim = 1)
        return self.logsoftmax(y)

class MLUTNet_4(nn.Module):

    def __init__(self, D_in, H, D_out):
        super(MLUTNet_4, self).__init__()
        self.__name__ = 'MLUTNet_4'
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.drop = nn.Dropout(0.5)

        self.d_in = D_in

        self.sub_nn_block_1_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_1_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))

        self.sub_nn_block_1_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))

        self.sub_nn_block_1_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))

        self.sub_nn_block_2_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_2_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_2_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_2_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_3_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_3_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_3_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_3_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_4_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_4_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_4_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_4_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))

        self.sub_nn_block_5_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, D_out + 1))
                ]
            ))

        self.sub_nn_block_5_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, D_out))
                ]
            ))
        
        self.sub_nn_block_5_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, D_out + 1))
                ]
            ))

        self.sub_nn_block_5_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, D_out))
                ]
            ))

    def forward(self, x):
        x = x.view(-1, self.d_in * 4) 
        i1, i2, i3, i4 = torch.split(x, [self.d_in, self.d_in, self.d_in, self.d_in], dim = 1)
        a11 = self.sub_nn_block_1_1(i1)
        a12 = self.sub_nn_block_1_2(i2)
        a13 = self.sub_nn_block_1_3(i3)
        a14 = self.sub_nn_block_1_4(i4)
        a21 = self.sub_nn_block_2_1(a11)
        a22 = self.sub_nn_block_2_2(a11+a12)
        a23 = self.sub_nn_block_2_3(a12+a13)
        a24 = self.sub_nn_block_2_4(a13+a14)
        a31 = self.sub_nn_block_3_1(a21+a22)
        a32 = self.sub_nn_block_3_2(a22+a23)
        a33 = self.sub_nn_block_3_3(a23+a24)
        a34 = self.sub_nn_block_3_4(a24)
        a41 = self.sub_nn_block_4_1(a31)
        a42 = self.sub_nn_block_4_2(a31+a32)
        a43 = self.sub_nn_block_4_3(a32+a33)
        a44 = self.sub_nn_block_4_4(a33+a34)
        a51 = self.sub_nn_block_5_1(a41+a42)
        a52 = self.sub_nn_block_5_2(a42+a43)
        a53 = self.sub_nn_block_5_3(a43+a44)
        a54 = self.sub_nn_block_5_4(a44)
        y = torch.cat((a51, a52, a53, a54), dim = 1)
        return self.logsoftmax(y)

class B_MLUTNet_4(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(B_MLUTNet_4, self).__init__()
        self.__name__ = 'B_MLUTNet_4'
        self.logsoftmax = nn.LogSoftmax(dim=1) # remember add dim=X when define
        self.drop = nn.Dropout(0.5)
        self.d_in = D_in

        self.sub_nn_block_1_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_1_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))

        self.sub_nn_block_1_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))

        self.sub_nn_block_1_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(D_in, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))

        self.sub_nn_block_2_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_2_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_2_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_2_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_3_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_3_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_3_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_3_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_4_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_4_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_4_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))
        
        self.sub_nn_block_4_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, H)),
                    ("bn1", nn.BatchNorm1d(H)),
                    ("tanh1", nn.Hardtanh())
                ]
            ))

        self.sub_nn_block_5_1 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, D_out + 1))
                ]
            ))

        self.sub_nn_block_5_2 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, D_out))
                ]
            ))
        
        self.sub_nn_block_5_3 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, D_out + 1))
                ]
            ))

        self.sub_nn_block_5_4 = nn.Sequential(
            OrderedDict(
                [
                    ("input", BinarizeLinear(H, D_out))
                ]
            ))

    def forward(self, x):
        x = x.view(-1, self.d_in * 4) 
        i1, i2, i3, i4 = torch.split(x, [self.d_in, self.d_in, self.d_in, self.d_in], dim = 1)
        a11 = self.sub_nn_block_1_1(i1)
        a12 = self.sub_nn_block_1_2(i2)
        a13 = self.sub_nn_block_1_3(i3)
        a14 = self.sub_nn_block_1_4(i4)
        a21 = self.sub_nn_block_2_1(a11)
        a22 = self.sub_nn_block_2_2(a11+a12)
        a23 = self.sub_nn_block_2_3(a12+a13)
        a24 = self.sub_nn_block_2_4(a13+a14)
        a31 = self.sub_nn_block_3_1(a21+a22)
        a32 = self.sub_nn_block_3_2(a22+a23)
        a33 = self.sub_nn_block_3_3(a23+a24)
        a34 = self.sub_nn_block_3_4(a24)
        a41 = self.sub_nn_block_4_1(a31)
        a42 = self.sub_nn_block_4_2(a31+a32)
        a43 = self.sub_nn_block_4_3(a32+a33)
        a44 = self.sub_nn_block_4_4(a33+a34)
        a51 = self.sub_nn_block_5_1(a41+a42)
        a52 = self.sub_nn_block_5_2(a42+a43)
        a53 = self.sub_nn_block_5_3(a43+a44)
        a54 = self.sub_nn_block_5_4(a44)
        y = torch.cat((a51, a52, a53, a54), dim = 1)
        return self.logsoftmax(y)

if args.dataset == 'mnist':
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'kmnist':
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
    datasets.KMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
    datasets.KMNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'fashionmnist':
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

def train(epoch):

    if epoch % 10 == 0:
        torch.cuda.empty_cache()
        if args.LR_mode == 'Decay':
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)
        output = model(data)
        training_loss = loss_func(output, target)

        optimizer.zero_grad()
        training_loss.backward()

        if model.__name__ == 'BNN' or model.__name__ == 'B_MLUTNet': # BNN or B-MLUTNet
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)

        optimizer.step()

        if model.__name__ == 'BNN' or model.__name__ == 'B_MLUTNet': # BNN or B-MLUTNet
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), training_loss))
    # loss recording
    writer.add_scalar('Training Loss', training_loss, global_step=epoch, walltime=None)  

def test():
    # Testing
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            test_loss += loss_func(output, target).item() # sum up batch loss
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)

    print('\nTest Result: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # Loss recording
    writer.add_scalar('Test Loss', test_loss, global_step=epoch, walltime=None)
    writer.add_scalar('Accuracy', correct / len(test_loader.dataset), global_step=epoch, walltime=None)

    # save model checkpoint
    if epoch == args.epochs:
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss,
        'accuracy': correct / len(test_loader.dataset)
        }, './checkpoints/{}-{}-{}-{}-epochs{}.pth'.format(args.dataset, model.__name__, optimizer.__class__.__name__,args.LR_mode, epoch))

    # save best performance checkpoint
    # if test_loss < best_loss:
    #     best_loss = test_loss
    #     best_epoch = epoch
    #     torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': test_loss
    #     }, './checkpoints/{}-best-exp-{}-epochs-{}-loss-{}.pth'.format(data_type, args.exp_number, epoch, loss))

# get prediction data for confusion matrix
@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    if args.cuda:
        all_preds = all_preds.cuda()
    for batch in loader:
        data, target = batch
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        preds = model(data)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

if __name__ == "__main__":

    # Choose models
    if args.exp_number == 1:
        model = DNN(args.input_size, args.hidden_size, args.output_size)
    elif args.exp_number == 2:
        model = BNN(args.input_size, args.hidden_size, args.output_size)
    elif args.exp_number == 3:
        model = MLUTNet(args.input_size // 2, args.hidden_size // 2, args.output_size // 2)
    elif args.exp_number == 4:
        model = B_MLUTNet(args.input_size // 2, args.hidden_size // 2, args.output_size // 2)
    elif args.exp_number == 5:
        model = MLUTNet_fc(args.input_size // 2, args.hidden_size // 2, args.output_size // 2)
    elif args.exp_number == 6:
        model = B_MLUTNet_fc(args.input_size // 2, args.hidden_size // 2, args.output_size // 2)

    # Set path of saved Results by categories
    now = datetime.now()
    exp = 'runs/' + now.strftime('%Y-%m-%d') + '/' + args.dataset + '/' + model.__name__ + '_' + args.LR_mode
    writer = SummaryWriter(exp)

    # Set GPU
    if args.cuda:
        torch.cuda.set_device(0) 
        model.cuda()

    # Loss function, learning rate, optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.LR_mode == 'Decay':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 10, gamma= 0.5, last_epoch= -1)
    elif args.LR_mode == 'CosineAnn':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    elif args.LR_mode == 'CosineAnnWarm':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)

    # (optional) Resume from checkpoints
    if args.resume:
        try:
            checkpoint = torch.load(args.checkpointspath)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
        except IOError:
            # no model loaded
            print('Warning: No checkpoints can be loaded. Model will be trained initially.')
            start_epoch = 1
    else:
        start_epoch = 1

    # Train & Testing
    for epoch in range(start_epoch, args.epochs + 1):
        train(epoch)
        test()
        scheduler.step()

    # Visualization 模型结构可视化
    if args.cuda:
        dummy_input = torch.rand(28, 28).sign().cuda()
    else:   
        dummy_input = torch.rand(28, 28).sign()
    writer.add_graph(model, (dummy_input,))

    # Draw confusion matrix 绘制混淆矩阵
    if args.dataset == 'mnist':
        train_set = datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    elif args.dataset == 'kmnist':
        train_set = datasets.KMNIST('../data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    elif args.dataset == 'fashionmnist':
        train_set = datasets.FashionMNIST('../data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    
    with torch.no_grad():
        prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
        train_preds = get_all_preds(model, prediction_loader)

    if args.cuda:
        train_preds = train_preds.cpu()
    cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
    names = ('0','1','2','3','4','5','6','7','8','9')
    plt.figure(figsize=(11,11))
    plot_confusion_matrix(cm, names, filename = 'cm_{}.jpg'.format(args.dataset + '_' + model.__name__ + '_' + args.LR_mode))
    