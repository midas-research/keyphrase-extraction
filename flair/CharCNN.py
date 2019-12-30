#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:37:54 2019

@author: r17935avinash
"""

from torch import nn

# Input - [ n_words , 100 , 25 ]
# Output - [ n_words , 30 ]

class CharCNN(nn.Module):
    def __init__(self):
        super(CharCNN,self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.conv1d = nn.Conv1d(100,30,3,padding=1)
        self.tanh = nn.Tanh()
        self.maxpool1d = nn.MaxPool1d(24)
        
    def forward(self,x):
        x = self.dropout(x)
        x = self.conv1d(x)
        x = self.tanh(x)
        x = self.maxpool1d(x) 
        x = self.dropout(x)
        x = x.view(x.size()[0],-1)
        return x
    
    
cnn = CharCNN()
