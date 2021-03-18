# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:33:02 2020

@author: Administrator
"""


import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from util.ReviewDataset import ReviewDataset
from torch.utils.data import Dataset,DataLoader

def dataloader_build(num_labeled,total_ind,path_train,path_test,class_num):
    BATCH_SIZE=32
    
    #现将图片和标签读取出来
    ROOT_TRAIN = path_train 
    ROOT_TEST = path_test 
#    transform = transforms.Compose([transforms.CenterCrop(64),transforms.Grayscale(1),transforms.ToTensor()])
#    transform = transforms.Compose([transforms.CenterCrop(90),transforms.RandomHorizontalFlip(),transforms.RandomRotation(degrees=(0, 10)),transforms.Grayscale(1),transforms.ToTensor()])
    transform = transforms.Compose([
             transforms.CenterCrop(64),
             transforms.RandomVerticalFlip(),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(5),
             transforms.Grayscale(1),
            transforms.ToTensor()])#soc 用的data augamentation
#    transform = transforms.Compose([transforms.CenterCrop(90),transforms.Grayscale(1),transforms.ToTensor()])
    train_dataset = ImageFolder(ROOT_TRAIN, transform=transform)
    transform = transforms.Compose([transforms.CenterCrop(64),transforms.Grayscale(1),transforms.ToTensor()])
   
    test_dataset = ImageFolder(ROOT_TEST, transform=transform)
    
    #print(train_dataset[0]) # 训练集第一张图像张量以及对应的标签，二维元组
    #print(train_dataset[0][0]) # 训练集第一张图像的张量
    #print(train_dataset[0][1]) # 训练集第一张图像的标签
#    n = len(train_dataset) 
#    permute = np.random.permutation(n)#根据所有的标签数据的数量 建立不重复的随机数组 取一部分做标签数据
    permute = total_ind
    unlabeled_idx = permute[num_labeled:]#另外一部分做无标签数据
    labeled_idx = permute[:num_labeled]
    
    
    labeled_data = [train_dataset[i][0] for i in labeled_idx]                                             
    labeled_labels = [train_dataset[i][1] for i in labeled_idx] 
#    print(labeled_labels)
    unlabeled_data = [train_dataset[i][0] for i in unlabeled_idx]                                                
    unlabeled_labels_ture = [train_dataset[i][1] for i in unlabeled_idx]                                          
    unlabeled_labels = [-1 for _ in range(len(unlabeled_data))]#给所有的无标签数据的标签位填上-1
    train_data = [train_dataset[i][0] for i in permute]                                             
    train_labels = [train_dataset[i][1] for i in permute] 
    
    m = len(test_dataset)
    test_idx = np.arange(m)
    
    
    test_data = [test_dataset[i][0] for i in test_idx]                                             
    test_labels = [test_dataset[i][1] for i in test_idx] 
    print("Creating dataloaders......")

                                              
    w_list = [1. for i in range(len(labeled_data))]
    c_list = [1. for i in range(class_num)]
    train_dataset = ReviewDataset(labeled_data, labeled_labels, w_list, c_list, 128)
    
    train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,num_workers=0)
    train_loader_shuffle = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,num_workers=0)
    
    w_list = [1. for i in range(len(train_labels))]
    c_list = [1. for i in range(class_num)]
    groundtruth_dataset = ReviewDataset(train_data, train_labels, w_list, c_list, 128)
    groundtruth_loader = DataLoader(dataset=groundtruth_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,num_workers=0)
    
    w_list = [1. for i in range(len(test_data))]
    c_list = [1. for i in range(class_num)]
    val_dataset = ReviewDataset(test_data, test_labels, w_list, c_list, 128)
    val_loader = DataLoader(dataset=val_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,num_workers=0)
    
    w_list = [1. for i in range(len(unlabeled_labels_ture))]
    c_list = [1. for i in range(class_num)]
    unlabeled_dataset = ReviewDataset(unlabeled_data, unlabeled_labels_ture, w_list, c_list, 128)
    unlabeled_loader = DataLoader(dataset=unlabeled_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,num_workers=0)
    print("Building dataloaders done!")                                                
                                                    
#    class_names = train_dataset.classes
#    print('class:',class_names)
#    N_class=len(class_names)
#    print(N_class)
    d = {
        "unlabeled_idx": unlabeled_idx,
        "labeled_idx": labeled_idx,
        "train_loader": train_loader,
        "groundtruth_loader": groundtruth_loader,
        "all_data": train_data,  # all images in train
        "groundtruth_labels": train_labels,
        "val_loader":val_loader,
        "train_loader_shuffle":train_loader_shuffle,
        "unlabeled_loader":unlabeled_loader
        }
    return d