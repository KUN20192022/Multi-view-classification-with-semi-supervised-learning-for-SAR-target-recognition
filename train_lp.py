# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:27:18 2020

@author: Administrator
"""
import numpy as np
import util.Net
from util.data_build import dataloader_build
from util.train_model import *
from util.fea_ext import *
from util.lp import *
from util.pre_function import *
def train_LP_model(PATH1,PATH2,PATH3,data,num_labeled,total_epochs,path_train,path_test):
    # PATH1是基础模型 PATH2是最终模型 PATH3是提取特征模型 num_labeled 是有标签的数量 total_epochs是训练总数
    # data 是生成的对应的索引
    
    total_ind = data['total_ind']        
    total_ind =np.squeeze(total_ind ) 
   
#    d = dataloader_build(num_labeled,total_ind)
    N_class = 10
    k=3
    d = dataloader_build(num_labeled,total_ind,path_train,path_test,N_class) 
    batch_features = feature_extractor1(d["groundtruth_loader"],PATH3)
    p_labels, updated_weights, updated_class_weights = label_propagation(batch_features, d["groundtruth_labels"], d["labeled_idx"], d["unlabeled_idx"],N_class, k=k)
    pseudo_loader = update_pseudoloader(d["all_data"], p_labels, updated_weights, updated_class_weights)

    print("Epoch 0")

    train_model_phase1(pseudo_loader,d["val_loader"],1,PATH1,PATH2)

    # epoch 1-T'
#% LP 阶段
    for i in range(total_epochs):
        print("Epoch {}".format(i + 1))
        d = dataloader_build(num_labeled,total_ind,path_train,path_test,N_class) 
        batch_features = feature_extractor1(d["groundtruth_loader"],PATH3)
        p_labels, updated_weights, updated_class_weights = label_propagation(batch_features, d["groundtruth_labels"], d["labeled_idx"], d["unlabeled_idx"],N_class, k=k)
        pseudo_loader = update_pseudoloader(d["all_data"], p_labels, updated_weights, updated_class_weights)

        
        train_model_phase1(pseudo_loader,d["val_loader"],1,PATH2,PATH2)