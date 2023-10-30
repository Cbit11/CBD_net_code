import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os
from torch.utils.data import dataset, DataLoader
import shutil
import random
import torch.optim


def transform(x):
    x= torch.tensor(x, dtype= torch.float32)/255.0
    x= x.reshape(x.shape[2], 600,800)
    return x

class Dataset:
    def __init__(self, gt, noisy):
        self.gt= gt
        self.noisy= noisy
    def __len__(self):
        return len(self.gt)
    def __getitem__(self, idx):
        gt_pth= self.gt[idx]
        noisy_pth= self.noisy[idx]
        gt= cv2.imread(gt_pth)
        noisy= cv2.imread(noisy_pth)
        gt= cv2.resize(gt,(600,800))
        noisy= cv2.resize(noisy,(600, 800))
        gt= transform(gt)
        noisy= transform(noisy)
        return gt, noisy

train_size= int(len(GT)*0.8)
test_size= int(len(GT)*0.2)
gt_train= []
gt_test=[]
noisy_train=[]
noisy_test=[]
for i, j in enumerate(GT):
    if i< train_size:
        gt_train.append(j)
    else:
        gt_test.append(j)

for i, j in enumerate(noisy):
    if i< train_size:
        noisy_train.append(j)
    else:
        noisy_test.append(j)