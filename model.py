import torch
import torch.nn as nn
import numpy as np 

def conv_block(in_channel, out_channel, kernel_size,stride=1, padding=0):
    lis= nn.Sequential(nn.Conv2d(in_channel, out_channel,kernel_size,stride, padding), nn.ReLU())
    return lis

#noise estimation net
class noise_estimation_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1= conv_block(3,32,3,1,1)
        self.conv= nn.ModuleList([conv_block(32,32,3,1,1) for i in range(3)])
        self.conv2= conv_block(32,3,3,1,1)
    def forward(self, x):
        x1= self.conv1(x)
        for i , j in enumerate(self.conv):
            x1= self.conv[i // 2](x1) + j(x1)
        out= self.conv2(x1)
        return out

class subNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1= conv_block(6,64, kernel_size=(3,3), stride= 1, padding = 1)
        self.conv2= conv_block(64,64,(3,3),1,1)
        self.conv3= conv_block(64,128, (3,3),1,1)
        
        self.conv4= conv_block(128,128, (3,3),1,1)
        self.conv5= conv_block(128,128, (3,3),1,1)
        
        self.conv6= conv_block(128, 256, (3,3),1,1)
        self.conv7= conv_block(256,256, (3,3),1,1)
        self.conv8= conv_block(256,256, (3,3),1,1)
        self.conv9= conv_block(256,256, (3,3),1,1)
        self.conv10= conv_block(256,256, (3,3),1,1)
        self.conv11= conv_block(256,256, (3,3),1,1)
        
        self.transconv1= nn.ConvTranspose2d(256, 128, (3,3), stride=2, padding = 1)
        self.conv12= conv_block(128, 128, (3,3),1,1)
        self.conv13= conv_block(128, 128, (3,3),1,1)
        self.conv14= conv_block(128, 128, (3,3),1,1)
        
        self.transconv2= nn.ConvTranspose2d(128,64, (3,3), stride=2, padding = 1)
        
        self.conv15= conv_block(64,64, (3,3),1,1)
        self.conv16= conv_block(64,64, (3,3),1,1)
        
        self.conv17= nn.Conv2d(64, 3, (1,1), stride =1,padding =1)
        self.act= nn.ReLU()
    def forward(self, x):
        x1= self.conv1(x)
        x1= self.conv2(x1)
        pool1= nn.AvgPool2d((2,2))(x1)
        x2= self.conv3(pool1)
        x2= self.conv4(x2)
        x2= self.conv5(x2)
        pool2= nn.AvgPool2d((2,2))(x2)
        x3= self.conv6(pool2)
        x3= self.conv7(x3)
        x3= self.conv8(x3)
        x3= self.conv9(x3)
        x3= self.conv10(x3)
        x3= self.conv11(x3)
        upsample1= self.act(self.transconv1(x3))
        upsample1= nn.functional.pad(upsample1, (0,1,0,1))
        x4= upsample1+ x2
        x4= self.conv12(x4)
        x4= self.conv13(x4)
        x4= self.conv14(x4)
        upsample2= self.act(self.transconv2(x4))
        upsample2= nn.functional.pad(upsample2, (0,1,0,1))
        x5= upsample2+x1
        x5= self.conv15(x5)
        x5= self.conv15(x5)
        out= self.conv17(x5)
        return out

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_estimation= noise_estimation_net()
        self.subnet= subNet()
    def forward(self, x):
        x1= self.noise_estimation(x)
        x2= torch.cat((x, x1),1)
        x3= self.subnet(x2)
        out = x3[:,:,:-2,:-2]
        out = out+ x
        return out