# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:20:22 2020

@author: Administrator
"""
import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import copy
import time
from util.Net import Net
import os
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import scipy.io as io
import numpy as np


#%%
def train_model_base(train_loader,val_loader,num_epoch,PATH1):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    device = torch.device("cpu")
#    net.load_state_dict(torch.load('E:/2019phd_simulation/20200415MMD/soc_half_base50epoch.pkl'))
#    best_model_wts = copy.deepcopy(net.state_dict())
    net=Net()
    net.cuda()
    net.train()
    
    learning_rate=0.01
    
    criterion = nn.CrossEntropyLoss()
   
    since = time.time()##计时
    running_loss=[]
    training_accuracy=[]
    val_accuracy=[]
    best_acc=0
 
    for epoch in range(num_epoch):#  num_epoch
        correct = 0
        total = 0
        loss_total=0.0
        lr =  learning_rate * (0.1 ** (epoch // 50))
        optimizer = optim.SGD(net.parameters() , lr  , momentum = 0.9)
        
        for i , (data_batch,labels_batch,w,c) in enumerate(train_loader):###第二个参数为该函数打印标号的初始值，默认从0开始打印，该函数返回一个enumerate类型的数据。
            
#            inputs , labels = data
            data_batch  = Variable(data_batch.to(device)) 
            labels_batch = Variable(labels_batch.to(device))
            w = Variable(w,requires_grad=True).to(device)
            c = Variable(c,requires_grad=True).to(device)
            minibatch_size = len(labels_batch)
            optimizer.zero_grad()
            outputs=net(data_batch.to(device))[0]
            
            loss=criterion(outputs , labels_batch)
            loss=loss*w.type(torch.cuda.FloatTensor)
            loss=loss*c.type(torch.cuda.FloatTensor)
            loss=loss.sum()/minibatch_size
         
            loss=loss.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item()) 

            _ , predicted = torch.max(outputs.data , 1)
            correct += (predicted == labels_batch.to(device)).sum().item()
         
            total += labels_batch.size(0)
            loss_total += loss.item()
            if i % 2 == 1:
                loss_total =loss_total / 2
                print('[%d , %2d] loss: %.5f' % (epoch + 1 , i + 1 , loss_total))
                ####每一次的20个轮回包括20*batch_size个图片训练加载的loss
                
                loss_total = 0.0
        
        epoch_acc= 100*correct / total
        
        print('Accuracy of the network on the train images: %.3f %%'%  epoch_acc )
        
        training_accuracy.append(epoch_acc)
        correct=0
        total=0
#不需要验证        
        for i , (data_batch,labels_batch,w,c) in enumerate(val_loader): 
            data_batch = data_batch.to(device)
            labels_batch = labels_batch.to(device)
#            w = Variable(w,requires_grad=True).to(device)
#            c = Variable(c,requires_grad=True).to(device)
            outputs=net(data_batch.to(device))[0]
            _ , predicted = torch.max(outputs.data , 1)
            correct += (predicted == labels_batch.to(device)).sum().item()
            total += labels_batch.size(0)
        epoch_acc=100 * correct/total
        val_accuracy.append(epoch_acc)
        if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())


#    best_model_wts = copy.deepcopy(net.state_dict())
    torch.save(net.state_dict(best_model_wts), PATH1)
#            
#        print('Accuracy of the network on the val images: %.3f %%' % epoch_acc )
    print('Finished Base Training')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))###运行时间
    return
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

#%%
def train_model_base_eoc(train_loader,val_loader,num_epoch,PATH1,PATH2):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    device = torch.device("cpu")
    
    
    net=Net()
    net.cuda()
#    net.load_state_dict(torch.load(r'E:\2019phd_simulation\20200713SSLEM_gpu\model\eocmodel\eoc45_100c.pkl'))
    net.train()
    best_model_wts = copy.deepcopy(net.state_dict())
    class_names=['2S1','BRDN_2','ZSU_23/4']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters() , lr = 0.0001 , momentum = 0.9)
    since = time.time()##计时
    running_loss=[]
    training_accuracy=[]
    val_accuracy=[]
    best_acc=0
    final_prob=[]
    final_pre=[]
    final_labels=[]
    for epoch in range(num_epoch):#  num_epoch
        correct = 0
        total = 0
        loss_total=0.0
      
   
        for i , (data_batch,labels_batch,w,c) in enumerate(train_loader):###第二个参数为该函数打印标号的初始值，默认从0开始打印，该函数返回一个enumerate类型的数据。
            
#            inputs , labels = data
            data_batch  = Variable(data_batch.to(device)) 
            labels_batch = Variable(labels_batch.to(device))
            w = Variable(w,requires_grad=True).to(device)
            c = Variable(c,requires_grad=True).to(device)
            minibatch_size = len(labels_batch)
            optimizer.zero_grad()
            outputs=net(data_batch.to(device))[0]
            
            loss=criterion(outputs , labels_batch)
            loss=loss*w.type(torch.cuda.FloatTensor)
            loss=loss*c.type(torch.cuda.FloatTensor)
            loss=loss.sum()/minibatch_size
         
            loss=loss.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item()) 

            _ , predicted = torch.max(outputs.data , 1)
            correct += (predicted == labels_batch.to(device)).sum().item()
         
            total += labels_batch.size(0)
            loss_total += loss.item()
            if i % 2 == 1:
                loss_total =loss_total / 2
                print('[%d , %2d] loss: %.5f' % (epoch + 1 , i + 1 , loss_total))
                ####每一次的20个轮回包括20*batch_size个图片训练加载的loss
                
                loss_total = 0.0
        
        epoch_acc= 100*correct / total
        
        print('Accuracy of the network on the train images: %.3f %%'%  epoch_acc )
        
        training_accuracy.append(epoch_acc)
        correct=0
        total=0

        pre_labels=[]
        total_labels=[]
       
    #            net.load_state_dict(best_model_wts)
        prob_outs=torch.zeros(1,3)      
#        for i , (data_batch,labels_batch,w,c) in enumerate(val_loader): 
#        
#            data_batch = data_batch.to(device)
#            
#            for k in labels_batch:
##                print(k)
#               
#                total_labels.append(k)
#    #            w = Variable(w,requires_grad=True).to(device)
#    #            c = Variable(c,requires_grad=True).to(device)
#            labels_batch = labels_batch.to(device)
#            outputs=net(data_batch.to(device))[0]
#            _ , predicted = torch.max(outputs.data , 1)
#                
#            outputs=outputs.detach().cpu().numpy()
#            prob_outs=np.vstack((prob_outs,outputs))
#            correct += (predicted == labels_batch.to(device)).sum().item()
#            total += labels_batch.size(0)
#            
#            for n in predicted:
#    #            print(predicted.type)
#                n=n.cpu()
#                pre_labels.append(n)
#           
#        epoch_testacc=100 * correct / total
#    
#        prob_outs=np.delete(prob_outs, 0, 0)
#        val_accuracy.append(epoch_testacc)
#        
#        if epoch_testacc > best_acc:
#                best_acc = epoch_testacc
#                best_model_wts = copy.deepcopy(net.state_dict())
#                final_prob=prob_outs
#                final_pre=pre_labels
#                final_labels=total_labels
##                print('update')
#        print(epoch_testacc)
#    final_prob=prob_outs
#    final_pre=pre_labels
#    final_labels=total_labels            
    best_model_wts = copy.deepcopy(net.state_dict())
#    houxuan=select_labels(final_prob)
#    results_cm(final_labels,final_pre,class_names)
#    result_save(final_prob,final_pre,best_acc,final_labels,houxuan,PATH2)

    
#    best_model_wts = copy.deepcopy(net.state_dict())
    torch.save(net.state_dict(best_model_wts), PATH1)
#            
#        print('Accuracy of the network on the val images: %.3f %%' % epoch_acc )
    print('Finished Base Training')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))###运行时间
    return best_acc,val_accuracy,final_prob,final_pre,final_labels
def train_model_base_eoc_phase1(train_loader,val_loader,num_epoch,PATH1,PATH3,PATH2,best,final_prob,final_pre,final_labels):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    device = torch.device("cpu")
    
    
    net=Net()
    net.cuda()
    net.load_state_dict(torch.load(PATH3))
    net.train()
    best_model_wts = copy.deepcopy(net.state_dict())
    class_names=['2S1','BRDN_2','ZSU_23/4']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters() , lr = 0.0001 , momentum = 0.9)
    since = time.time()##计时
    running_loss=[]
    training_accuracy=[]
    val_accuracy=[]
    best_acc= best
#    final_prob=[]
#    final_pre=[]
#    final_labels=[]
    for epoch in range(num_epoch):#  num_epoch
        correct = 0
        total = 0
        loss_total=0.0
      
   
        for i , (data_batch,labels_batch,w,c) in enumerate(train_loader):###第二个参数为该函数打印标号的初始值，默认从0开始打印，该函数返回一个enumerate类型的数据。
            
#            inputs , labels = data
            data_batch  = Variable(data_batch.to(device)) 
            labels_batch = Variable(labels_batch.to(device))
            w = Variable(w,requires_grad=True).to(device)
            c = Variable(c,requires_grad=True).to(device)
            minibatch_size = len(labels_batch)
            optimizer.zero_grad()
            outputs=net(data_batch.to(device))[0]
            
            loss=criterion(outputs , labels_batch)
            loss=loss*w.type(torch.cuda.FloatTensor)
            loss=loss*c.type(torch.cuda.FloatTensor)
            loss=loss.sum()/minibatch_size
         
            loss=loss.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item()) 

            _ , predicted = torch.max(outputs.data , 1)
            correct += (predicted == labels_batch.to(device)).sum().item()
         
            total += labels_batch.size(0)
            loss_total += loss.item()
            if i % 2 == 1:
                loss_total =loss_total / 2
                print('[%d , %2d] loss: %.5f' % (epoch + 1 , i + 1 , loss_total))
                ####每一次的20个轮回包括20*batch_size个图片训练加载的loss
                
                loss_total = 0.0
        
        epoch_acc= 100*correct / total
        
        print('Accuracy of the network on the train images: %.3f %%'%  epoch_acc )
        
        training_accuracy.append(epoch_acc)
        correct=0
        total=0

        pre_labels=[]
        total_labels=[]
       
    #            net.load_state_dict(best_model_wts)
        prob_outs=torch.zeros(1,3)      
        for i , (data_batch,labels_batch,w,c) in enumerate(val_loader): 
        
            data_batch = data_batch.to(device)
            
            for k in labels_batch:
#                print(k)
               
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
        val_accuracy.append(epoch_testacc)
        
        if epoch_testacc > best_acc:
                best_acc = epoch_testacc
                best_model_wts = copy.deepcopy(net.state_dict())
                final_prob=prob_outs
                final_pre=pre_labels
                final_labels=total_labels
#                print('update')
#        print(epoch_testacc)
            
    houxuan=[]
#    houxuan=select_labels(final_prob)
    results_cm(final_labels,final_pre,class_names)
    result_save(final_prob,final_pre,best_acc,final_labels,houxuan,PATH2)

    
#    best_model_wts = copy.deepcopy(net.state_dict())
    torch.save(net.state_dict(best_model_wts), PATH1)
#            
#        print('Accuracy of the network on the val images: %.3f %%' % epoch_acc )
    print('Finished Base Training')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))###运行时间
    return best_acc,val_accuracy,final_prob,final_pre,final_labels
#%%
def train_model_phase1(train_loader,val_loader,num_epoch,PATH1,PATH2):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net=Net().cuda()
#    device = torch.device("cpu")
    net.load_state_dict(torch.load(PATH1))
    best_model_wts = copy.deepcopy(net.state_dict())
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters() , lr = 0.001 , momentum = 0.9)
#    since = time.time()##计时
    running_loss=[]
    training_accuracy=[]
#    val_accuracy=[]
    best_acc=0
    for epoch in range(num_epoch):
        correct = 0
        total = 0
        loss_total=0.0
        for i , (data_batch,labels_batch,w,c) in enumerate(train_loader):###第二个参数为该函数打印标号的初始值，默认从0开始打印，该函数返回一个enumerate类型的数据。
            
#            inputs , labels = data
            data_batch  = Variable(data_batch.to(device)) 
            labels_batch = Variable(labels_batch.to(device))
            w = Variable(w,requires_grad=True).to(device)
            c = Variable(c,requires_grad=True).to(device)
            minibatch_size = len(labels_batch)
            optimizer.zero_grad()
            outputs=net(data_batch.to(device))[0]
            
            loss=criterion(outputs , labels_batch)
            loss=loss*w.type(torch.cuda.FloatTensor)
            loss=loss*c.type(torch.cuda.FloatTensor)
#            loss=loss*w.type(torch.FloatTensor)
#            loss=loss*c.type(torch.FloatTensor)
            loss=loss.sum()/minibatch_size
         
            loss=loss.type(torch.cuda.FloatTensor)
#            loss=loss.type(torch.FloatTensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item()) 

            _ , predicted = torch.max(outputs.data , 1)
            correct += (predicted == labels_batch.to(device)).sum().item()
            total += labels_batch.size(0)
            loss_total += loss.item()
            if i % 2 == 1:
                loss_total =loss_total /2
                print('[%d , %2d] loss: %.5f' % (epoch + 1 , i + 1 , loss_total))
                ####每一次的20个轮回包括20*batch_size个图片训练加载的loss
                
                loss_total = 0.0
        
        epoch_acc=100.0 * correct / total
        
        print('Accuracy of the network on the train images: %.3f %%' % epoch_acc )
        
        training_accuracy.append(epoch_acc)
                     
        correct=0
        total=0
        
        for i , (data_batch,labels_batch,w,c) in enumerate(val_loader):
            data_batch = data_batch.to(device)
            labels_batch = labels_batch.to(device)
#            w = Variable(w,requires_grad=True).to(device)
#            c = Variable(c,requires_grad=True).to(device)
            outputs=net(data_batch.to(device))[0]
            _ , predicted = torch.max(outputs.data , 1)
            correct += (predicted == labels_batch.to(device)).sum().item()
            total += labels_batch.size(0)
        epoch_acc=100 * correct/total
#        print('Accuracy of the network on the val_dataloader: %.3f %%' % epoch_acc )
#        val_accuracy.append(epoch_acc)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(net.state_dict())
#            print('Update the modsel')
#        else :
        if  os.path.exists(PATH2):
            os.remove(PATH2)
    
        
    #    best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(net.state_dict(best_model_wts), PATH2)
        
     
#            
#        print('Accuracy of the network on the val images: %.3f %%' % epoch_acc )
#    print('Finished Base Training')
#    time_elapsed = time.time() - since
#    print('Training complete in {:.0f}m {:.0f}s'.format(
#            time_elapsed // 60, time_elapsed % 60))###运行时间
#    print(best_acc)
    return 