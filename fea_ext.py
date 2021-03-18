# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:04:49 2020

@author: Administrator
"""
from util.Net import Net
import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable


#from utils import *
def feature_extractor1(data_loader,PATH1):
    net=Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.cuda()
#    device = torch.device("cpu")
#    net.load_state_dict(torch.load('E:/2019phd_simulation/Code_multiNMF/soc_experiment/model/soc0405100_BN.pkl'))
    net.load_state_dict(torch.load(PATH1))
    net.eval()

#            net.load_state_dict(best_model_wts)

    extra_features=torch.zeros(1,128)
    for i , (data_batch,labels_batch,w,c) in enumerate(tqdm(data_loader)):

        outputs = net(Variable(data_batch.to(device)))[0]
        features=net(data_batch.to(device))[1]
        features=features.detach().cpu().numpy()
        extra_features=np.vstack((extra_features,features))
        _ , predicted = torch.max(outputs.data , 1)

    extra_features=np.delete(extra_features, 0, 0)
#    print('Accuracy of the network on the test images: %.3f %%' % epoch_testacc)
#    test_accuracy.append(epoch_testacc)
#    results_cm(total_labels,pre_labels)
#    result_save(prob_outs,pre_labels,test_accuracy,total_labels,extra_features)
    return extra_features 