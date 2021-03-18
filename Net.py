# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:36:09 2020

@author: Administrator
"""

import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net , self).__init__()
        self.conv1 = nn.Conv2d(1 , 32 , 5,stride=1, padding=0)### 输入通道数为1，输出通道数为16，卷积核大小为5
        self.pool = nn.MaxPool2d(2 , 2)
        self.bn1= nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True)
        self.conv2 = nn.Conv2d(32 , 64 , 3, stride=1, padding=1)
        self.bn2= nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.conv3 = nn.Conv2d(64 , 128 , 3, stride=2, padding=1)
        self.bn3= nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.conv4 = nn.Conv2d(128 , 256 , 3, stride=2, padding=1)
        self.bn4= nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
        self.dropout1=nn.Dropout2d(0.1)
        
         
        
        self.fc1 = nn.Linear(256 * 1 * 1 , 128)
        self.fc2 = nn.Linear(128, 10)
#        self.fc3 = nn.Linear(256 , 10)
#            self.fc4 = nn.Linear(42 , 2)
  

 
    def forward(self , x):
        x = self.pool(F.relu(self.conv1(x)))
        x=self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x=self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x=self.bn3(x)
        f=self.pool(x)
        f = f.view(-1 , 128 * 2 * 2)
        x = self.pool(F.relu(self.conv4(x)))
        x=self.bn4(x)
        x=self.dropout1(x)
        
        x = x.view(-1 , 256 * 1 * 1)  #利用view函数使得conv2层输出的16*5*5维的特征图尺寸变为400大小从而方便后面的全连接层的连接
        x_features = F.relu(self.fc1(x))
#        fea = F.relu(self.fc2(x_features))
        
        return F.log_softmax(self.fc2(x_features), dim=1) ,x_features
    
class Net_test(nn.Module):
    def __init__(self):
        super(Net_test , self).__init__()
        self.conv1 = nn.Conv2d(1 , 32 , 5,stride=1, padding=0)### 输入通道数为1，输出通道数为16，卷积核大小为5
        self.pool = nn.MaxPool2d(2 , 2)
        self.bn1= nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True)
        self.conv2 = nn.Conv2d(32 , 64 , 3, stride=1, padding=1)
        self.bn2= nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.conv3 = nn.Conv2d(64 , 128 , 3, stride=2, padding=1)
        self.bn3= nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.conv4 = nn.Conv2d(128 , 256 , 3, stride=2, padding=1)
        self.bn4= nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
        self.dropout1=nn.Dropout2d(0.1)
        
         
        
        self.fc1 = nn.Linear(256 * 1 * 1 , 128)
        self.fc2 = nn.Linear(128, 10)
#        self.fc3 = nn.Linear(256 , 10)
#            self.fc4 = nn.Linear(42 , 2)
  


    def forward(self , x):
        x = self.pool(F.relu(self.conv1(x)))
        x=self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x=self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x=self.bn3(x)
        f=self.pool(x)
        f = f.view(-1 , 128 * 2 * 2)
        x = self.pool(F.relu(self.conv4(x)))
        x=self.bn4(x)
        x=self.dropout1(x)
        
        x = x.view(-1 , 256 * 1 * 1)  #利用view函数使得conv2层输出的16*5*5维的特征图尺寸变为400大小从而方便后面的全连接层的连接
        x_features = F.relu(self.fc1(x))
#        fea = F.relu(self.fc2(x_features))
        
        return F.softmax(self.fc2(x_features), dim=1) ,x_features
class Net1(nn.Module):
    def __init__(self):
        super(Net , self).__init__()
        self.conv1 = nn.Conv2d(1 , 16 , 5)### 输入通道数为1，输出通道数为16，卷积核大小为5
        self.pool = nn.MaxPool2d(2 , 2)
        self.bn1= nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True)
        self.conv2 = nn.Conv2d(16 , 64 , 6)
        self.bn2= nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.conv3 = nn.Conv2d(64 , 128 , 6)
        self.bn3= nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.conv4 = nn.Conv2d(128 , 256 , 6)
        self.bn4= nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
        self.dropout1=nn.Dropout2d(0.5)
        
         
        
        self.fc1 = nn.Linear(256 * 2 * 2 , 512)
        self.fc2 = nn.Linear(512 , 256)
        self.fc3 = nn.Linear(256 , 10)
#            self.fc4 = nn.Linear(42 , 2)
  


    def forward(self , x):
        x = self.pool(F.relu(self.conv1(x)))
        x=self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x=self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x=self.bn3(x)
        x = self.dropout1(F.relu(self.conv4(x)))
        x=self.bn4(x)
        
        
        x = x.view(-1 , 256 * 2 * 2)  #利用view函数使得conv2层输出的16*5*5维的特征图尺寸变为400大小从而方便后面的全连接层的连接
        x_features = F.relu(self.fc1(x))
        fea = F.relu(self.fc2(x_features))
        
        return F.log_softmax(self.fc3(fea), dim=1) ,x_features,fea 
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:27:30 2020

@author: Administrator
"""
import torch.nn.functional as F
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
 
 
# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
 
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
 
 
 
class ResNet(nn.Module):
    
    def __init__(self, block, layers, in_channels, out_channels):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.conv = conv3x3(1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
       
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(256, out_channels)
        
    def make_layer(self, block, out_channels, blocks, stride=1,mm=0):
        #print(out_channels,blocks,'****')
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
            
            
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        mm+=1
        
        self.in_channels = out_channels
        for i in range(1, blocks):
            
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
 
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
#        print(out.shape)
        out1 = F.softmax(self.fc(out), dim=1)
#        out1 = self.fc(out)
        return out1,out
 
 