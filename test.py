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
from model import network
from dataloader import Dataset


device= 'cuda' if torch.cuda.is_available() else 'cpu'
data_pth ="enter dataset path"
img_pth= []
for data in os.listdir(data_pth):
    for i in os.listdir(os.path.join(data_pth,data)):
        img_pth.append(data_pth+'/'+data+'/'+i)
lambd=2
GT=[]
noisy=[]
for i in range(len(img_pth)):
    if i%2==0:
        GT.append(img_pth[i])
    else:
        noisy.append(img_pth[i])

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
test_data= Dataset(gt_test, noisy_test)
test_loader= DataLoader(test_data, batch_size= 4)
mse= nn.MSELoss()
l1=nn.L1Loss()
model= network().to(device)
with torch.no_grad():
    for i,(gt,noisy) in enumerate(test_loader):
        gt= gt.to(device)
        noisy= noisy.to(device)
        denoised = model(noisy)
        loss= mse(denoised, gt)+ lambd*l1(denoised, gt)
        print('Loss {}'.format(loss.item()))