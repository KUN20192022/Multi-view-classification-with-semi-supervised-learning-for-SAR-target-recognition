# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:07:11 2020

@author: Administrator
"""

import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import scipy.io as io
import numpy as np
from util.Net import Net_test
from util.Net import ResNet

#%% print confusion matrix   
def results_cm(total_labels,predicted_best,class_names):
#    class_names=['0','1','2','3','4','5','6','7','8','9']
    labels_num=len(class_names)
    cm = confusion_matrix(total_labels,predicted_best,labels=range(labels_num))
    print(cm)
    n = len(cm)
    for r in range(len(cm[0])):
        colsum = sum(cm[r][i] for i in range(n))
        if cm[r][r]==0:
            print('Accuracy of %5s : %.3f %%' % (class_names[r],0))
        else:
            print ('Accuracy of %5s : %.3f %%' % (class_names[r],100*cm[r][r]/float(colsum)))
    return cm

#%%=================保存==============================================
def result_save(prob_outs,pre_labels,test_accuracy,total_labels,houxuan,PATH2):
    prob_outs_best=np.array(prob_outs)
    predicted_best=np.array(pre_labels)
#    running_loss=np.array(running_loss)
#    training_accuracy=np.array(training_accuracy)
    test_accuracy=np.array(test_accuracy)
#    val_accuracy=np.array(val_accuracy)
    total_labels=np.array(total_labels)
    houxuan=np.array(houxuan)
    io.savemat(PATH2,{'total_labels':total_labels,'prob_outs':prob_outs_best,'test_acc':test_accuracy,'predicted_best':predicted_best,'houxuan':houxuan})
    return  
#%%
def select_labels(data):
    hang_num=data.shape[0]
    # candidate_profile=np.zeros(hang_num,6)
    # candidate_profile=-1*candidate_profile
    #houxuan=[]
    aaa=[]
    # labels_4=torch.zeros(1,houxuan)
    for i in range(hang_num):
        multi_probas =data[i,:]
        K = np.array(multi_probas )
        proba_sorted_= sorted(K,reverse=True)
        top_probas_ = np.array(proba_sorted_[:10] )*100
    
        descend_ = (top_probas_[:-1] + 1/10)/(top_probas_[1:] + 1/10)
        aaa.append(np.argmax(descend_)+1) 
    return aaa

#%%testing
def test_model(dataloader,class_names,PATH1,PATH2):
       
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    net=Net_test().cuda()
    net=Net_test().cuda()
    net.eval()
    net.load_state_dict(torch.load(PATH1))
    N_class=len(class_names)
#    net.load_state_dict(torch.load(PATH1, map_location=lambda storage, loc: storage))
#    net.eval()
    correct = 0
    total = 0
    pre_labels=[]
    total_labels=[]
    test_accuracy=[]
#            net.load_state_dict(best_model_wts)
    prob_outs=torch.zeros(1,N_class)
    for data in dataloader:
        images , labels = data
        
        for k in labels:
           
            total_labels.append(k)
        outputs = net(Variable(images.to(device)))[0]
        
        
        _ , predicted = torch.max(outputs.data , 1)
        
        
        outputs=outputs.detach().cpu().numpy()
        prob_outs=np.vstack((prob_outs,outputs))
        correct += (predicted == labels.to(device)).sum().item()
        total += labels.size(0)
        
        for n in predicted:
#            print(predicted.type)
            n=n.cpu()
            pre_labels.append(n)
       
    epoch_testacc=100 * correct / total

    prob_outs=np.delete(prob_outs, 0, 0)
    print('Accuracy of the network on the test images: %.3f %%' % epoch_testacc)
    test_accuracy.append(epoch_testacc)
    houxuan=select_labels(prob_outs)
#    pre_labels=pre_labels.cpu()
   #    total_labels=total_labels.numpy()
#    total_labels=total_labels.tolist()
#    pre_labels=pre_labels.cpu().numpy()
#    pre_labels=pre_labels.tolist()
    results_cm(total_labels,pre_labels,class_names)
    result_save(prob_outs,pre_labels,test_accuracy,total_labels,houxuan,PATH2)
    return 
#%%
def test_model_eoc(dataloader,class_names,PATH1,PATH2):
       
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net=Net_test().cuda()
    net.eval()
    net.load_state_dict(torch.load(PATH1))
    N_class=len(class_names)
#    net.load_state_dict(torch.load(PATH1, map_location=lambda storage, loc: storage))
    net.eval()
    correct = 0
    total = 0
    pre_labels=[]
    total_labels=[]
    test_accuracy=[]
#            net.load_state_dict(best_model_wts)
    prob_outs=torch.zeros(1,N_class)
    
    for i , (data_batch,labels_batch,w,c) in enumerate(dataloader): 
        
        data_batch = data_batch.to(device)
        
        for k in labels_batch:
           
            total_labels.append(k)
#            w = Variable(w,requires_grad=True).to(device)
#            c = Variable(c,requires_grad=True).to(device)
        labels_batch = labels_batch.to(device)
        outputs=net(data_batch.to(device))[0]
        _ , predicted = torch.max(outputs.data , 1)
            
        outputs=outputs.detach().cpu().numpy()
        prob_outs=np.vstack((prob_outs,outputs))
        correct += (predicted == labels_batch.to(device)).sum().item()
        total += labels_batch.size(0)
        
        for n in predicted:
#            print(predicted.type)
            n=n.cpu()
            pre_labels.append(n)
       
    epoch_testacc=100 * correct / total

    prob_outs=np.delete(prob_outs, 0, 0)
    print('Accuracy of the network on the test images: %.3f %%' % epoch_testacc)
    test_accuracy.append(epoch_testacc)
    houxuan=select_labels(prob_outs)
#    print(total_labels)
#    print(pre_labels)
#    pre_labels=pre_labels.cpu()
   #    total_labels=total_labels.numpy()
#    total_labels=total_labels.tolist()
#    pre_labels=pre_labels.cpu().numpy()
#    pre_labels=pre_labels.tolist()
    results_cm(total_labels,pre_labels,class_names)
    result_save(prob_outs,pre_labels,test_accuracy,total_labels,houxuan,PATH2)
    return 
     
    
    
    
    
    
    
    
 