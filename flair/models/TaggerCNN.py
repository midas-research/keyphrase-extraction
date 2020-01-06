#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 22:03:23 2019

@author: r17935avinash
"""

# Structure
##inputs [n_sentences,n_words,embedding_length]
#                 | 
#                CNN
#                 |
##outputs [n_sentences, n_words, n_tags]

from torch import nn
import torch.nn.functional as F
import torch      
class Text_CNN(nn.Module):
    def __init__(self,in_c,out_c,max_length):
        super(Text_CNN,self).__init__()
        self.conv1 = nn.Conv1d(in_c,out_c,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(out_c,out_c,kernel_size=3,padding=1)
        self.Maxpool = nn.MaxPool1d(kernel_size=2)
        self.Dropout = nn.Dropout(p=0.5)
        self.max_length = max_length
        self.relu = nn.ReLU()
        
    def forward(self,x):
        (a,b,c) = x.size()
        x = F.pad(input=x,pad=(0,0,0,self.max_length-x.size(1),0,0))
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.Maxpool(x)
        x = self.Dropout(x)
        x = x.permute(0,2,1)[:,:b,:]
        return x
        
class Multi_Channel_CNN(nn.Module):
    def __init__(self,in_c,out_c,max_length):
        super(Multi_Channel_CNN,self).__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_c, out_c, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_c, out_c, kernel_size=2)
        self.Maxpool = nn.MaxPool1d(kernel_size=2)
        self.Linear = nn.Linear(3*out_c,out_c)
        self.Dropout = nn.Dropout(p=0.5)
        self.max_length = max_length
        self.relu = nn.ReLU()
        
    def forward(self,x):
        (a,b,c) = x.size()
        x = F.pad(input=x,pad=(0,0,0,self.max_length-x.size(1),0,0))
        x = x.permute(0,2,1)
        x = self.Dropout(x)
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x2 = self.conv2(x)
        x2 = self.relu(x2)
        x3 = self.conv3(x)
        x3 = self.relu(x3)
        x3 = F.pad(input=x3,pad=(0,1,0,0,0,0))
        m_x1 = self.Maxpool(x1)
        m_x2 = self.Maxpool(x2)
        m_x3 = self.Maxpool(x3)  
        x = torch.cat((m_x1,m_x2),dim=1)
        x = torch.cat((x,m_x3),dim=1)
        x = self.Dropout(x)
        x = x.permute(0,2,1)[:,:b,:]
        x = self.Linear(x)
        return x
    
class Preprocess_CNN(nn.Module):
    def __init__(self,in_c,out_c,max_length):
        super(Preprocess_CNN,self).__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)
        self.Maxpool = nn.MaxPool1d(kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.max_length = max_length       
        
    def forward(self,x):
        (a,b,c) = x.size()
        x = F.pad(input=x,pad=(0,0,0,self.max_length-x.size(1),0,0))
        x = x.permute(0,2,1)  
        x = self.conv1(x)
        x = self.relu(x)    
        x = self.dropout(x)
        x = x.permute(0,2,1)[:,:b,:]
        return x
        

            
        
    