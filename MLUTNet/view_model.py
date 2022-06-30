
from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d,BinarizeLinear_nobias
from models.binarized_modules import  Binarize,HingeLoss

# Pytorch保存自定义模型的办法：
# 不要使用torch.save，而是torch.save(model.state_dict(),'xxx.pkl')
# 在需要读取模型的时候，在新文件中声明并建立与原模型一致的class，再使用modelname.load_state_dict(torch.load('xxx.pkl'))

class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        self.infl_ratio=3
        num_of_hiddenlayers = 256
        self.fc1 = BinarizeLinear(784, num_of_hiddenlayers*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(num_of_hiddenlayers*self.infl_ratio)
        self.fc2 = BinarizeLinear(num_of_hiddenlayers*self.infl_ratio, num_of_hiddenlayers*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(num_of_hiddenlayers*self.infl_ratio)
        self.fc3 = BinarizeLinear(num_of_hiddenlayers*self.infl_ratio, num_of_hiddenlayers*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(num_of_hiddenlayers*self.infl_ratio)
        self.fc4 = nn.Linear(num_of_hiddenlayers*self.infl_ratio, 10)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)

class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.infl_ratio=1
        num_of_hiddenlayers = 128
        self.fc1 = BinarizeLinear_nobias(10, num_of_hiddenlayers*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        # self.bn1 = nn.BatchNorm1d(num_of_hiddenlayers*self.infl_ratio)
        self.fc2 = BinarizeLinear_nobias(num_of_hiddenlayers*self.infl_ratio, num_of_hiddenlayers*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        # self.bn2 = nn.BatchNorm1d(num_of_hiddenlayers*self.infl_ratio)
        self.fc3 = BinarizeLinear_nobias(num_of_hiddenlayers*self.infl_ratio, num_of_hiddenlayers*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        # self.bn3 = nn.BatchNorm1d(num_of_hiddenlayers*self.infl_ratio)
        self.fc4 = BinarizeLinear_nobias(num_of_hiddenlayers*self.infl_ratio, 10)
        # self.logsoftmax=nn.LogSoftmax()
        # self.drop=nn.Dropout(0.5)
        self.htanh4 = nn.Hardtanh()

    def forward(self, x):
        x = x.view(-1, 10)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        # x = self.drop(x)
        # x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        x = self.htanh4(x)
        return x

model1 = Net1()
model2 = Net2()


model1.load_state_dict(torch.load('modelparam.pkl'))
model2.load_state_dict(torch.load('logreg_state_dict.pkl'))

#输出模型参数以供查看
'''
print("MNIST BNN Model's state_dict:")
for param_tensor in model1.state_dict():
    print(param_tensor, "\t", model1.state_dict()[param_tensor].size())
    print(model1.state_dict()[param_tensor])
'''

print("LOGIC REGRESSION BNN Model's state_dict:")
for param_tensor in model2.state_dict():
    print(param_tensor, "\t", model2.state_dict()[param_tensor].size())
    print(model2.state_dict()[param_tensor])

#载入测试数据并且查看预测结果
X = np.loadtxt('test_in.txt')
X = torch.tensor(X[0:10, :])
X = X.float()

print('Predictions:')
print(model2(X))
