# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:32:24 2020

@author: Administrator
"""
import os
import scipy.io as sio
from util.ReviewDataset import ReviewDataset
from util.Net import Net
from util.data_build import dataloader_build
from util.train_model import * 
from util.train_lp import *
from util.fea_ext import *
from util.test_model_v2 import * 
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
#%%
def base_train():
    #    soc dataset
    path_train=r'E:/data/SOC/train'
    path_test=r'E:/data/SOC/test'
   
    data = sio.loadmat('./mat/index/label_index20_3.mat')     
    PATH1='./model/basemodel/base20b_50.pkl'#base model ,20 means percentage of train data, the last means epochs of base model

#    PATH1=r'E:\2019phd_simulation\20200808 Net update\checkpoints\model_2_4'
    PATH2='./model/basemodel/base20d_200.pkl'#base model ,20 means percentage of train data, the last means epochs of base model
#    PATH2='./model/LP20.pkl'
#    PATH3='./model/LP20_100.pkl'
#    PATH3='./model/shi1.pkl'    
    num_labeled=550
    N_class= 10
    total_ind = data['total_ind']   
    total_ind =np.squeeze(total_ind ) 
    d = dataloader_build(num_labeled,total_ind,path_train,path_test,N_class)   
    
    
    k=3
    total_epochs=150
#    train_model_base(d["train_loader_shuffle"],d["val_loader"],1,PATH1)
    train_model_phase1(d["train_loader_shuffle"],d["val_loader"],1,PATH1,PATH2)
    for i in range(total_epochs):
        print("Epoch {}".format(i + 1))
        d = dataloader_build(num_labeled,total_ind,path_train,path_test,N_class) 
#    net=Net().cuda()
        train_loader=d["train_loader_shuffle"]
        train_model_phase1(train_loader,train_loader,1,PATH2,PATH2)
        
#        train_LP_model(PATH1,PATH3,PATH3,data,num_labeled,total_epochs)  
    return PATH1,PATH2
#%%
def LP_train():
    path_train=r'E:/MSTAR_EXAMPLE/soc_ten_classes/train'
    path_test=r'E:/MSTAR_EXAMPLE/soc_ten_classes/test'
   
    data = sio.loadmat('./mat/index/label_index20_3.mat')   
    
#    PATH3=r'E:\2019phd_simulation\20200808 Net update\checkpoints\model_2_4'
    PATH1='./model/base20b_shiyixia.pkl'#base model ,20 means percentage of train data, the last means epochs of base model
    PATH2='./model/lpmodel/LP20b_shi.pkl'#base model ,20 means percentage of train data, the last means epochs of base model
#    PATH2='./model/LP20.pkl'
    PATH3='./model/base20b_50.pkl'
#    PATH3='./model/shi1.pkl'    
    num_labeled=550
    total_ind = data['total_ind']   
    total_ind =np.squeeze(total_ind ) 
      
    
    N_class= 10 
    d = dataloader_build(num_labeled,total_ind,path_train,path_test,N_class) 
    k=3
    total_epochs=1
#    train_model_base(d["train_loader_shuffle"],d["val_loader"],50,PATH1)#train a basic model
    train_model_phase1(d["train_loader_shuffle"],d["val_loader"],1,PATH1,PATH2)

    train_LP_model(PATH1,PATH2,PATH3,data,num_labeled,total_epochs,path_train,path_test)  
    return PATH1,PATH2
#%%
def final_test(path1,path2):
    batch_size=64


#transform = transforms.Compose([transforms.CenterCrop(90),transforms.ToTensor() , transforms.Normalize((0.5 , 0.5,0.5 ) , (0.5 , 0.5, 0.5 ))])  #前面参数是均值，后面是标准差
#    transform = transforms.Compose([transforms.CenterCrop(64),transforms.Grayscale(1),transforms.ToTensor()])  #前面参数是均值，后面是标准差,
#no transformation when test using the trained model
    transform = transforms.Compose([
             transforms.CenterCrop(64),
             transforms.RandomVerticalFlip(),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(10),
             
             transforms.Grayscale(1),
            transforms.ToTensor()])
###transform to tensor 已经将每个元素/255，进行了归一化也就是转化为了0-1，后面的0.5是标准化至-1到1
    trainset = torchvision.datasets.ImageFolder(root='E:/MSTAR_EXAMPLE/soc_ten_classes/train' ,  transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset , batch_size = batch_size , shuffle = False )
    dataset_sizes = len(trainset) 
    print(dataset_sizes)
    transform = transforms.Compose([transforms.CenterCrop(64),transforms.Grayscale(1),transforms.ToTensor()])
    testset = torchvision.datasets.ImageFolder(root = 'E:/MSTAR_EXAMPLE/soc_ten_classes/test' , transform = transform)
#    testset = torchvision.datasets.ImageFolder(root = 'E:/EMtest/EOC1test/eoc1/test_45' , transform = transform)
    testloader = torch.utils.data.DataLoader(testset , batch_size = batch_size , shuffle = False )
    class_names = trainset.classes
    print('class:',class_names)
    N_class=len(class_names)
    print(N_class)
    test_model(trainloader,class_names,path1,path2)
#    test_model(testloader,class_names,path1,path2)
    
    
      
    
    
    
    
#%%

if __name__ =='__main__':
#    base_train()
#    LP_train()main.py
    path_model_base='./model/lpmodel/LP20b_shi.pkl'
    path_save='yuan1.mat'
#    path_save='./mat/lpmat/LP20b_shi.mat'
    final_test(path_model_base,path_save)
    
    
#    path_model_base='E:/2019phd_simulation/20200421LPSSL/model/20LP/LP_20_110epoch_jun_rota2.pkl'
#    path_save='yuan1.mat'
#    final_test(path_model_base,path_save)
# %%   
#    path_model_lp=main()[1]
#    path_save='./mat/LP20_100.mat'
#    final_test(path_model_lp,path_save)