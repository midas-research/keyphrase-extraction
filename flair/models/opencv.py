#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:37:03 2020

@author: r17935avinash
"""

import cv2

path = "/Users/r17935avinash/Downloads/F1@5_extractive.png"
img = cv2.imread(path)
cv2.imshow('image',img)
