"""
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
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F


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



class VanillaNet_2(nn.Module):
    def __init__(self, n_c=1, n_f=32, leak=0.2, p =0, scaling=1):
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
        self.scaling = scaling
        
    def forward(self, x):       
      x = x.view(-1, self.n_c, 32, 32)
      
      return self.scaling * self.f(x).squeeze()


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



class NonlocalNet(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.2, scaling=1):
        super(NonlocalNet, self).__init__()
        self.n_c = n_c
        self.n_f = n_f
        self.scaling = scaling
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=n_c, out_channels=n_f, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),

            NonLocalBlock(in_channels=n_f),
            nn.Conv2d(in_channels=n_f, out_channels=n_f * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),

            NonLocalBlock(in_channels=n_f * 2),
            nn.Conv2d(in_channels=n_f * 2, out_channels=n_f * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=(n_f * 4) * 4 * 4, out_features=n_f * 8),
            nn.LeakyReLU(negative_slope=leak),
            nn.Linear(in_features=n_f * 8, out_features=1)
        )

    def forward(self, x):
        x = x.view(-1, self.n_c, 32, 32)
        conv_out = self.convs(x).view(x.shape[0], -1)
        conv_out = conv_out * self.scaling
        return self.fc(conv_out).squeeze()



def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, dim_out, 1),
    )

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.proj(x)
        #x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
   
    def __init__(self, dim, dim_out, *, groups=8):
        super().__init__()
        
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
       
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)



# structure of non-local block (from Non-Local Neural Networks https://arxiv.org/abs/1711.07971)
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, sub_sample=True):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // 2)

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = t.matmul(theta_x, phi_x)
        f_div_c = F.softmax(f, dim=-1)

        y = t.matmul(f_div_c, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        w_y = self.W(y)
        z = w_y + x

        return z
    
    

class NonlocalNet_Big(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.2, scaling=1):
        super(NonlocalNet_Big, self).__init__()
        self.n_c = n_c
        self.n_f = n_f
        self.scaling = scaling
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=n_c, out_channels=n_f, kernel_size=3, stride=1, padding=1),

            ResnetBlock(n_f, n_f),
            ResnetBlock(n_f, n_f),
            NonLocalBlock(in_channels=n_f),
            Downsample(n_f, n_f*2),

            ResnetBlock(n_f*2, n_f*2),
            ResnetBlock(n_f*2, n_f*2),
            NonLocalBlock(in_channels=n_f*2),
            Downsample(n_f*2, n_f*4),

            ResnetBlock(n_f*4, n_f*4),
            ResnetBlock(n_f*4, n_f*4),
            NonLocalBlock(in_channels=n_f*4),
            Downsample(n_f*4, n_f*8),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=(n_f * 8) * 4 * 4, out_features=n_f * 8),
            nn.LeakyReLU(negative_slope=leak),
            nn.Linear(in_features=n_f * 8, out_features=1)
        )

    def forward(self, x):
        x = x.view(-1, self.n_c, 32, 32)
        conv_out = self.convs(x).view(x.shape[0], -1)
        conv_out = conv_out * self.scaling
        return self.fc(conv_out).squeeze()

# Define Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Define Conv2D layer with Swish activation
class Conv2dSwish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2dSwish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.swish = Swish()

    def forward(self, x):
        return self.swish(self.conv(x))

# Define Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dSwish(channels, channels, 3, 1, 1)
        self.conv2 = Conv2dSwish(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

# EBM Architecture
class EBM_Big(nn.Module):
    def __init__(self, n_f=64, n_c=3, scaling=1):
        super(EBM_Big, self).__init__()
        self.n_c =n_c
        self.n_f = n_f
        self.conv1 = Conv2dSwish(self.n_c, self.n_f, 3, 1, 1)
        self.res1 = ResidualBlockArchitecture(self.n_f, self.n_f * 2)
        self.res2 = ResidualBlockArchitecture(self.n_f*2, self.n_f * 4)
        self.res3 = ResidualBlockArchitecture(self.n_f*4, self.n_f * 8)
        self.conv2 = Conv2dSwish(self.n_f * 8, 100, 3, 4, 0)

    def forward(self, x):
        x = x.view(-1,self.n_c,32,32)
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.conv2(x)
        x = x.view(x.shape[0],-1)
        return x.sum(axis=1)
        #return x.sum(dim=1)  # Sum over channel dimension

# Residual Block Architecture
class ResidualBlockArchitecture(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockArchitecture, self).__init__()
        self.conv1 = Conv2dSwish(in_channels, out_channels, 3, 1, 1)
        self.conv2 = Conv2dSwish(out_channels, out_channels, 3, 1, 1)
        self.conv3 = Conv2dSwish(out_channels, out_channels, 3, 1, 1)
        self.avg_pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        residual = self.conv1(x)
        x = self.conv2(residual)
        x = self.conv3(x) + residual
        return self.avg_pool(x)

####################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_f=32):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, n_f, 3, padding=1)  # Output: 16x32x32
        self.pool = nn.AvgPool2d(2, 2)  # Output after pooling: 16x16x16
        self.conv2 = nn.Conv2d(n_f, n_f * 2, 3, padding=1) # Output: 32x16x16, after pooling: 32x8x8
        self.conv3 = nn.Conv2d(n_f * 2, n_f * 4, 3, padding=1) # Output: 64x8x8, after pooling: 64x4x4

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.pool(x)
        return x

class Decoder(nn.Module):
    def __init__(self, n_f=32):
        super(Decoder, self).__init__()
        self.convTrans1 = nn.ConvTranspose2d(n_f*4, n_f*2, 2, stride=2) # Upsampling to 32x8x8
        self.conv1 = nn.Conv2d(n_f*2, n_f*2, 3, padding=1)  # Processing
        self.convTrans2 = nn.ConvTranspose2d(n_f*2, n_f, 2, stride=2) # Upsampling to 16x16x16
        self.conv2 = nn.Conv2d(n_f, n_f, 3, padding=1)  # Processing
        self.convTrans3 = nn.ConvTranspose2d(n_f, 3, 2, stride=2)  # Upsampling to 3x32x32
        self.conv3 = nn.Conv2d(3, 3, 3, padding=1)   # Final processing

    def forward(self, x):
        x = self.convTrans1(x)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.convTrans2(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.convTrans3(x)
        x = self.conv3(x)  # Using sigmoid for the output layer
        return x

class Autoencoder(nn.Module):
    def __init__(self, n_f=64, n_c=3, scaling=1):
        super(Autoencoder, self).__init__()
        self.n_f = n_f
        self.n_c = n_c
        self.encoder = Encoder(n_f)
        self.decoder = Decoder(n_f)

    def forward(self, x):
        x = x.view(-1,self.n_c, 32 ,32)
        y = x
        y = self.encoder(y)
        y = self.decoder(y)
        batch_size = x.shape[0]
        y = y.view(batch_size, -1)
        x = x.view(batch_size, -1)
        return ((y - x) **2).sum(axis=1)

# Model instantiation
#autoencoder = Autoencoder().cuda()
#print(autoencoder(torch.randn(64,3,32,32).cuda()).shape)

"""


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
    def __init__(self, n_c=1, n_f=32, leak=0.2, p =0, scaling=1):
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
            nn.Conv2d(n_f*8, n_f*16, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Dropout(p),
            nn.Conv2d(n_f*16, 1, 4, 1, 0))
        self.log_T = nn.Parameter(t.tensor(1)*-9.90348755254)
        self.T = nn.Parameter(t.tensor(5e-5))
        self.n_c = n_c
        self.conv1 = nn.Conv2d(n_c, n_f, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_f, n_f * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(n_f*2, n_f*4, 4, 2, 1)
        self.conv4 = nn.Conv2d(n_f*4, n_f*8, 4, 2, 1)
        self.conv5 = nn.Conv2d(n_f*8, 1, 4, 1, 0)
        self.scaling = scaling
        
    def forward(self, x):       
      x = x.view(-1, self.n_c, 64, 64)
      
      return self.scaling * self.f(x).squeeze()


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



class NonlocalNet(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, scaling=1):
        super(NonlocalNet, self).__init__()

        self.n_c = n_c
        self.n_f = n_f
        self.scaling = scaling
        self.log_T = nn.Parameter(t.tensor(1)*-2.3)
        self.T = nn.Parameter(t.tensor(1e-1))

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=n_c, out_channels=n_f, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),

            NonLocalBlock(in_channels=n_f),
            nn.Conv2d(in_channels=n_f, out_channels=n_f * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),

            NonLocalBlock(in_channels=n_f * 2),
            nn.Conv2d(in_channels=n_f * 2, out_channels=n_f * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=(n_f * 4) * 4 * 4, out_features=n_f * 8),
            nn.LeakyReLU(negative_slope=leak),
            nn.Linear(in_features=n_f * 8, out_features=1)
        )

    def forward(self, x):
        x = x.view(-1, self.n_c, 32, 32)
        conv_out = self.convs(x).view(x.shape[0], -1)
        conv_out = self.fc(conv_out).squeeze()
        x = x.view(x.shape[0], -1)
        #conv_out = conv_out + 1e-1*(x**2).sum(axis=1)
        return conv_out

# structure of non-local block (from Non-Local Neural Networks https://arxiv.org/abs/1711.07971)
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, sub_sample=True):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // 2)

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = t.matmul(theta_x, phi_x)
        f_div_c = F.softmax(f, dim=-1)

        y = t.matmul(f_div_c, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        w_y = self.W(y)
        z = w_y + x
        return z
    
    


# Define Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Define Conv2D layer with Swish activation
class Conv2dSwish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2dSwish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.swish = Swish()

    def forward(self, x):
        return self.swish(self.conv(x))

# Define Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dSwish(channels, channels, 3, 1, 1)
        self.conv2 = Conv2dSwish(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

# EBM Architecture
class EBM_Big(nn.Module):
    def __init__(self, n_f=64, n_c=3, scaling=1):
        super(EBM_Big, self).__init__()
        self.n_c =n_c
        self.n_f = n_f
        self.conv1 = Conv2dSwish(self.n_c, self.n_f, 3, 1, 1)
        self.res1 = ResidualBlockArchitecture(self.n_f, self.n_f * 2)
        self.res2 = ResidualBlockArchitecture(self.n_f*2, self.n_f * 4)
        self.res3 = ResidualBlockArchitecture(self.n_f*4, self.n_f * 8)
        self.conv2 = Conv2dSwish(self.n_f * 8, 100, 3, 4, 0)

    def forward(self, x):
        x = x.view(-1,self.n_c,32,32)
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.conv2(x)
        x = x.view(x.shape[0],-1)
        return x.sum(axis=1)
        #return x.sum(dim=1)  # Sum over channel dimension

# Residual Block Architecture
class ResidualBlockArchitecture(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockArchitecture, self).__init__()
        self.conv1 = Conv2dSwish(in_channels, out_channels, 3, 1, 1)
        self.conv2 = Conv2dSwish(out_channels, out_channels, 3, 1, 1)
        self.conv3 = Conv2dSwish(out_channels, out_channels, 3, 1, 1)
        self.avg_pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        residual = self.conv1(x)
        x = self.conv2(residual)
        x = self.conv3(x) + residual
        return self.avg_pool(x)



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
      y = (x.view(x.shape[0],-1)**2).sum(axis=1) + self.f(x).squeeze()
      return y
    
class ToyNet(nn.Module):
    def __init__(self, dim=100, n_c=1, n_f=64, leak=0.05, p =0, scaling=1):
        super(ToyNet, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(dim, n_f, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, n_f * 2, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, n_f * 2, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, 1, 1, 1, 0))

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return self.f(x).squeeze()

class SimpleEnergyNet(nn.Module):
    def __init__(self, hidden_dim=256,n_c=1, n_f=32, leak=0.2, p =0, scaling=1):
        super().__init__()
        # First conv layer: 10 channels -> hidden_dim channels
        self.conv1 = nn.Conv2d(10, hidden_dim, kernel_size=1)
        # Second conv layer: hidden_dim -> hidden_dim
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)    
        # Third conv layer: hidden_dim -> 1 (energy value)
        self.conv3 = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.T =nn.Parameter(t.tensor(1)*5.01e-3)
    def forward(self, x):
        # Input shape: (batch, 10, 1, 1)
        # First conv + ReLU
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.leaky_relu(self.conv1(x))
        # Second conv + ReLU
        x = self.leaky_relu(self.conv2(x))
        # Final conv to get energy value and squeeze all size-1 dimensions
        return self.conv3(x).squeeze()
        #x = x.view(x.shape[0], -1)
        #return abs((x**2).sum(axis=1) - 1)


class SelfAttnBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            batch_first=True,
            dropout=dropout
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):                    # (B,10,d_model)
        y = self.attn(x, x, x, need_weights=False)[0]
        x = self.ln1(x + y)
        y = self.ff(x)
        x = self.ln2(x + y)
        return x

# ---------------------------------------------------------
#  Energy network with 3 self‑attention layers
# ---------------------------------------------------------
class AttnEnergyNet(nn.Module):
    """
    • accepts (B,10,1,1)  or  (B,10) tensors
    • no hand‑crafted features
    • three attention blocks learn any interactions they need
    • after the blocks, average the 10 token embeddings and
      map to one scalar energy
    """
    def __init__(self, d_model=64, n_head=4, n_layers=3, dropout=0.0, n_c=1, n_f=32, leak=0.2, p =0, scaling=1):
        super().__init__()
        self.token_embed = nn.Linear(1, d_model)          # scalar → vector
        self.blocks = nn.ModuleList([
            SelfAttnBlock(d_model, n_head, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1)                 # pooled → energy

    def forward(self, x):   
        x = x.view(x.shape[0], x.shape[1], 1, 1)                              # (B,10,1,1)  or (B,10)
        if x.ndim == 4:                                   # flatten dummy H,W
            x = x.squeeze(-1).squeeze(-1)                 # (B,10)
        x = x.unsqueeze(-1)                               # (B,10,1)
        x = self.token_embed(x)                           # (B,10,d_model)
        for blk in self.blocks:
            x = blk(x)                                    # (B,10,d_model)
        x = x.mean(dim=1)                                 # global mean pool
        energy = self.head(x).squeeze(-1)                 # (B,)
        return energy
    


import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math
"""
# ----------------------------------
# 1) ResBlock: two 3×3 Convs + residual
# ----------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout):
        super(ResBlock, self).__init__()
        self.norm1    = nn.GroupNorm(8, in_ch)
        self.act1     = nn.SiLU()
        self.conv1    = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2    = nn.GroupNorm(8, out_ch)
        self.act2     = nn.SiLU()
        self.dropout  = nn.Dropout(dropout)
        self.conv2    = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        if self.res_conv is not None:
            x = self.res_conv(x)
        return x + h

# ----------------------------------
# 2) SpatialSelfAttention at a given resolution
# ----------------------------------
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_ch, heads, head_dim):
        super(SpatialSelfAttention, self).__init__()
        assert in_ch == heads * head_dim, "in_ch must = heads × head_dim"
        self.norm    = nn.GroupNorm(8, in_ch)
        self.qkv     = nn.Conv2d(in_ch, in_ch * 3, kernel_size=1)
        self.proj    = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.heads   = heads
        self.head_dim = head_dim

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        # reshape for multi-head: (b, heads, head_dim, h*w)
        q = q.view(b, self.heads, self.head_dim, h*w).permute(0,1,3,2)  # (b,heads,N,head_dim)
        k = k.view(b, self.heads, self.head_dim, h*w)                  # (b,heads,head_dim,N)
        v = v.view(b, self.heads, self.head_dim, h*w).permute(0,1,3,2)  # (b,heads,N,head_dim)

        attn = (q @ k) * (1.0 / math.sqrt(self.head_dim))  # (b,heads,N,N)
        attn = F.softmax(attn, dim=-1)
        out  = attn @ v                                    # (b,heads,N,head_dim)
        out  = out.permute(0,1,3,2).contiguous().view(b, c, h, w)
        return x + self.proj(out)

# ----------------------------------
# 3) UNet -> outputs head_channels × 32 × 32
# ----------------------------------
class UNet(nn.Module):
    def __init__(self,
                 n_c=3,
                 base_channels=128,
                 channel_mult=[1,2,2,2],
                 num_res_blocks=2,
                 attention_resolution=16,
                 attention_heads=4,
                 head_channels=64*4,
                 dropout=0.1):
        super(UNet, self).__init__()
        self.n_c = n_c

        # initial 3×3 conv
        self.init_conv = nn.Conv2d(n_c,
                                   base_channels * channel_mult[0],
                                   kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # helper to build one level of ResBlocks (+ optional attention)
        def make_level(in_ch, out_ch, level):
            layers = []
            for _ in range(num_res_blocks):
                layers.append(ResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
            if 32 // (2**level) == attention_resolution:
                layers.append(SpatialSelfAttention(out_ch, attention_heads, head_channels // attention_heads))
            return nn.Sequential(*layers)

        # encoder levels
        self.enc1 = make_level(base_channels * channel_mult[0], base_channels * channel_mult[0], level=0)  # 32×32
        self.enc2 = make_level(base_channels * channel_mult[0], base_channels * channel_mult[1], level=1)  # 16×16
        self.enc3 = make_level(base_channels * channel_mult[1], base_channels * channel_mult[2], level=2)  # 8×8
        self.enc4 = make_level(base_channels * channel_mult[2], base_channels * channel_mult[3], level=3)  # 4×4

        # bottleneck @ 4×4
        self.bottleneck = nn.Sequential(*[
            ResBlock(base_channels * channel_mult[3],
                     base_channels * channel_mult[3],
                     dropout)
            for _ in range(num_res_blocks)
        ])

        # decoder: upsample + ResBlocks
        self.up3  = nn.ConvTranspose2d(base_channels * channel_mult[3],
                                       base_channels * channel_mult[2],
                                       kernel_size=2, stride=2)
        self.dec3 = make_level(base_channels * (channel_mult[2] + channel_mult[3]),
                               base_channels * channel_mult[2],
                               level=2)

        self.up2  = nn.ConvTranspose2d(base_channels * channel_mult[2],
                                       base_channels * channel_mult[1],
                                       kernel_size=2, stride=2)
        self.dec2 = make_level(base_channels * (channel_mult[1] + channel_mult[2]),
                               base_channels * channel_mult[1],
                               level=1)

        self.up1  = nn.ConvTranspose2d(
            base_channels * channel_mult[1],  # 256
            base_channels * channel_mult[0],  # 128
            kernel_size=2, stride=2
        )

        # the only change is here:
        self.dec1 = make_level(
            base_channels * (channel_mult[0] + channel_mult[0]),  # 128 + 128 = 256
            base_channels * channel_mult[0],                      # 128
            level=0
        )

        # final conv to head_channels
        self.final_conv = nn.Conv2d(base_channels * channel_mult[0],
                                    head_channels,
                                    kernel_size=3, padding=1)

    def forward(self, x):
        # enforce shape
        x = x.view(-1, self.n_c, 32, 32)

        x1 = self.init_conv(x)
        e1 = self.enc1(x1);  x = self.pool(e1)
        e2 = self.enc2(x);   x = self.pool(e2)
        e3 = self.enc3(x);   x = self.pool(e3)
        e4 = self.enc4(x)

        x = self.bottleneck(e4)

        x = self.up3(x);     x = torch.cat([x, e3], dim=1); x = self.dec3(x)
        x = self.up2(x);     x = torch.cat([x, e2], dim=1); x = self.dec2(x)
        x = self.up1(x);     x = torch.cat([x, e1], dim=1); x = self.dec1(x)

        x = self.final_conv(x)
        return x

# ----------------------------------
# 4) Transformer (ViT) head → scalar energy
# ----------------------------------
class TransformerHead(nn.Module):
    def __init__(self,
                 in_channels,
                 patch_size=4,
                 embed_dim=384,
                 num_layers=8,
                 num_heads=4,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 output_scale=1000.0):
        super(TransformerHead, self).__init__()

        # patch‐embedding via Conv2d
        self.patch_embed = nn.Conv2d(in_channels,
                                     embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        num_patches = (32 // patch_size) ** 2

        # learnable [CLS] token + positional embeddings
        self.cls_token = nn.Parameter(t.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(t.zeros(1, num_patches + 1, embed_dim))
        self.dropout   = nn.Dropout(dropout)

        # standard TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final projection to scalar
        self.norm = nn.LayerNorm(embed_dim)
        self.fc   = nn.Linear(embed_dim, 1)
        self.output_scale = output_scale

        # init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: [B, in_channels, 32,32]
        b = x.shape[0]
        x = self.patch_embed(x)                 # [B, embed_dim, 8,8]
        x = x.flatten(2).transpose(1, 2)        # [B, 64, embed_dim]
        cls = self.cls_token.expand(b, -1, -1)  # [B, 1, embed_dim]
        x = t.cat([cls, x], dim=1)              # [B, 65, embed_dim]
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer expects [seq_len, batch, embed_dim]
        x = x.transpose(0, 1)
        x = self.transformer(x)                 # [65, B, embed_dim]
        cls_out = x[0]                          # [B, embed_dim]
        cls_out = self.norm(cls_out)
        energy  = self.fc(cls_out).squeeze(-1)  # [B]
        return energy * self.output_scale

# ----------------------------------
# 5) Full EBM: UNet → TransformerHead → Energy scalar
# ----------------------------------
class UNetTransformerEBM(nn.Module):
    def __init__(self, dropout=0.0, n_c=1, n_f=32, leak=0.2, p =0, scaling=1):
        super(UNetTransformerEBM, self).__init__()

        # 1) UNet backbone (→ 256 × 32 × 32)
        self.unet = UNet(
            n_c=3,
            base_channels=128,
            channel_mult=[1,2,2,2],
            num_res_blocks=2,
            attention_resolution=16,
            attention_heads=4,
            head_channels=4*64,   # 4 heads × 64 channels/head = 256
            dropout=0.1
        )

        # 2) ViT‐style head
        self.transformer_head = TransformerHead(
            in_channels=256,     # matches head_channels above
            patch_size=4,
            embed_dim=384,
            num_layers=8,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.1,
            output_scale=1000.0
        )

    def forward(self, x):
        # x: [B,3,32,32]
        x = x.view(-1, 3, 32, 32)
        feats  = self.unet(x)                      # [B,256,32,32]
        energy = self.transformer_head(feats)      # [B]
        return energy
"""

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, leak=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act1 = nn.LeakyReLU(leak)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act2 = nn.LeakyReLU(leak)

        # Shortcut connection (1x1 conv if dimensions differ)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act1(self.conv1(x))
        out = self.conv2(out)
        out = residual + out
        return self.act2(out)

class ToyNetResidual(nn.Module):
    def __init__(self, dim=100, n_c=1, n_f=64, leak=0.2, p=0, scaling=1):
        super(ToyNetResidual, self).__init__()
        self.initial = nn.Conv2d(dim, n_f, 1, 1, 0)

        self.block1 = ResidualBlock(n_f, n_f * 2, leak)
        self.block2 = ResidualBlock(n_f * 2, n_f * 2, leak)
        self.block3 = ResidualBlock(n_f * 2, n_f * 2, leak)

        self.final = nn.Conv2d(n_f * 2, 1, 1, 1, 0)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final(x)
        return x.squeeze()


class ResidualBlockSN(nn.Module):
    def __init__(self, in_channels, out_channels, leak=0.2):
        super(ResidualBlockSN, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
        self.act1 = nn.LeakyReLU(leak)
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0))
        self.act2 = nn.LeakyReLU(leak)

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act1(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.act2(out)

class ToyNetResidualSN(nn.Module):
    def __init__(self, dim=100, n_c=1, n_f=64, leak=0.2, p=0, scaling=1):
        super(ToyNetResidualSN, self).__init__()
        self.initial = spectral_norm(nn.Conv2d(dim, n_f, 1, 1, 0))

        self.block1 = ResidualBlockSN(n_f, n_f * 2, leak)
        self.block2 = ResidualBlockSN(n_f * 2, n_f * 2, leak)
        self.block3 = ResidualBlockSN(n_f * 2, n_f * 2, leak)

        self.final = spectral_norm(nn.Conv2d(n_f * 2, 1, 1, 1, 0))

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final(x)
        return x.squeeze()
