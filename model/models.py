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
        #x = x.view(x.shape[0], -1)
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
