import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import matplotlib.pyplot as plt
import math
import torch as t
from torch.nn import Softplus
from torch.nn import Softplus
from torch.nn.utils import spectral_norm

class T(nn.Module):
    def __init__(self, n_c=1, n_f=32, leak=0.2, p =0,bias=True):
        super(T, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1,bias=bias),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1,bias=bias),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1,bias=bias),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1,bias=bias),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*8, n_f*16, 4,bias=bias)
            )
        
        self.linear = nn.Linear(256,1)

    def forward(self, x):

        x = x.view(-1,1,32,32)

        x = self.f(x).squeeze()

        return x



class VanillaNet(nn.Module):
    def __init__(self, n_c=1, n_f=32, leak=0.2, p =0):
        super(VanillaNet, self).__init__()
        self.n_c = n_c
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),  # 16x16
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),  # 8x8
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),  #4x4
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*8, n_f*16, 4, 2, 1))  #2x2
        
        self.theta = nn.Linear(n_f*16*2*2,1)
        self.T = nn.Linear(n_f*16*2*2,1)
        self.W =  nn.Parameter(1e-2*torch.randn(32*32*n_c,n_f*16*2*2))
                              
                             
    def forward(self, x):
        x = x.view(-1,self.n_c,32,32)
        y = self.f(x)
        y = y.view(y.shape[0],-1)
        E = self.theta(y)
        return E



class Temperature(nn.Module):
    
    def __init__(self):
        super(Temperature, self).__init__()
        self.log_T = nn.Parameter((t.tensor(1).double())*-9.90348755254)
        self.T = nn.Parameter(t.tensor(1)*5e-5)

"""
class Temperature(nn.Module):
    def __init__(self, n_c=1, n_f=32, leak=0.2, p =0):
        super(Temperature, self).__init__()
        self.n_c = n_c
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),  # 16x16
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),  # 8x8
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),  #4x4
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*8, n_f*16, 4, 2, 1))  #2x2
        
        self.theta = nn.Linear(n_f*16*2*2,1)
        self.W =  nn.Parameter(1e-2*torch.randn(32*32*n_c,n_f*16*2*2))
                              
                             
    def forward(self, x):
        x = x.view(-1,self.n_c,32,32)
        y = self.f(x)
        y = y.view(y.shape[0],-1)
        y = self.theta(y)
        return y
        """
class VanillaNet_2(nn.Module):
    def __init__(self, n_c=1, n_f=32, leak=0.2, p =0):
        super(VanillaNet_2, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(leak),
            nn.Dropout(p),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Dropout(p),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Dropout(p),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Dropout(p),
            nn.Conv2d(n_f*8, 1, 4, 1, 0))
        self.log_T = nn.Parameter(t.tensor(1)*-9.90348755254)
        self.T = nn.Parameter(t.tensor(5e-5))
        self.n_c = n_c
        self.conv1 = nn.Conv2d(n_c, n_f, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_f, n_f * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(n_f*2, n_f*4, 4, 2, 1)
        self.conv4 = nn.Conv2d(n_f*4, n_f*8, 4, 2, 1)
        self.conv5 = nn.Conv2d(n_f*8, 1, 4, 1, 0)

    
    def forward(self, x):
        
      x = x.view(-1,self.n_c,32,32)

      return self.f(x).squeeze()


class VanillaNet_2_Spectral(nn.Module):
    def __init__(self, n_c=1, n_f=32, leak=0.2, p =0):
        super(VanillaNet_2_Spectral, self).__init__()
        self.f = nn.Sequential(
            spectral_norm(nn.Conv2d(n_c, n_f, 3, 1, 1)),
            nn.LeakyReLU(leak),
            nn.Dropout(p),
            spectral_norm(nn.Conv2d(n_f, n_f * 2, 4, 2, 1)),
            nn.LeakyReLU(leak),
            nn.Dropout(p),
            spectral_norm(nn.Conv2d(n_f*2, n_f*4, 4, 2, 1)),
            nn.LeakyReLU(leak),
            nn.Dropout(p),
            spectral_norm(nn.Conv2d(n_f*4, n_f*8, 4, 2, 1)),
            nn.LeakyReLU(leak),
            nn.Dropout(p),
            spectral_norm(nn.Conv2d(n_f*8, 1, 4, 1, 0)),
            nn.Tanh())
        self.n_c = n_c
        self.Temp = nn.Parameter(torch.ones(1))

    def forward(self, x):
      x = x.view(-1,self.n_c,32,32)

      #y = (x.view(x.shape[0],-1)**2).sum(axis=1) + self.f(x).squeeze()

      #return y
      return self.f(x).squeeze()
