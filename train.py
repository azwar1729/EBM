import numpy as np
import math
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.transforms as tr
from torchvision.utils import make_grid , save_image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
from torch.nn.utils import spectral_norm
import torchvision.utils as tv_utils
from PIL import Image
from torch.nn import Softplus
import torch as t
from tqdm import tqdm
import torch as t
import shutil
import os
import argparse
import sys
import json
from model.sampler import sample_HMC, sample_Langevin
from model import models
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
from utils_jgm.tikz_pgf_helpers import tpl_save
import utils
import trial
from tqdm import tqdm

def train(config,root_path,resume_checkpoint=False):
    
    logger = utils.set_logger(root_path+"logging.log")
    #logger.info("WORKING!!")
    
    ########################  Get Data ###################################
    
    data = {'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
            'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
            'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func)}

    transform = tr.Compose([tr.Resize(config['im_sz']),
                            tr.ToTensor()])
    q = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform)]).cuda()
    
        
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data',
        train=True,
        download = True,
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize(32)])
        ),
        batch_size=64
    )
    #################### Intialize model/Optimizer #######################
    if resume_checkpoint==False:
        model = getattr(models,config['model'])(n_c=config['im_ch'])
        model = model.cuda()
        optimizer= optim.Adam(model.parameters(),lr=1e-4)
        train_iter = 0
    
    else:
        checkpoint = torch.load(root_path + f'/state.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_iter = checkpoint['iteration']
    
    ############################ Training #################################
    
    logger.info("Training Started ...")
    for ii in range(train_iter,config['num_train_iters']):
    #for ii,data in enumerate(tqdm(train_loader)):
        # Get training data, reshape and add noise
        x_data  = q[ torch.randperm(q.shape[0])[0:config['batch_size']] ] 
        x_data = x_data.reshape(x_data.shape[0],-1)
        x_data = x_data + config['data_noise'] * torch.randn_like(x_data)
        
        # Intialize starting point of the Langevin chain
        if config['init'] == "noise":
            x_init = torch.randn_like(x_data)
        elif config['data'] == "data":
            x_init = x_data
        elif config['data'] == "persistent":
            x_init = x_bank ## FILL IT LATER 

        # Set Temperature parameter as required
        if config["T"] == "learnable_T":
            T = model.T
        elif config["T"] == "learnable_log_T":
            T = torch.exp(model.log_T)
        elif not isinstance(config["T"],torch.Tensor):   # If Temperature is not tensor then convert to tensor
            T = t.tensor(config["T"])
        
        # Run Langevin Dynamics
        x_sample,acceptance_ratio, energy, score, acceptance_components,transition = sample_Langevin(x_init, model,L=config['L'],eps=config['eps'],T=T,MH=config["MH"],transition_steps=config['transition_steps'])
        
        loss = model(x_data).mean() - model(x_sample).mean()
        if config['scale_loss'] == "True" or isinstance(config["T"],str) == True:
            print("HERE")
            loss = loss / T
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(ii,loss.item())
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        
        if ii%config["logging_freq"] == 0:
             
            utils.plot_multiple_images(x_sample.view(-1,config['im_ch'],config['im_sz'],config['im_sz']),root_path + f"sample_{ii}.png")
            #utils.plot_transition_images(transition,root_path + f"transition_{ii}.png")
            utils.plot_lineplot(acceptance_ratio,root_path + f"acceptance_ratio_{ii}.png")
            utils.plot_lineplot(energy,root_path + f"energy_{ii}.png")
            utils.plot_lineplot(score,root_path + f"score_{ii}.png")
            
            torch.save({
              'iteration': ii,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, root_path + f'/state-{ii}.pth')
        
            torch.save({
              'iteration': ii,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, root_path + f'/state.pth')
        
        if ii%config["long_run_freq"]==0:
            
            x_sample,acceptance_ratio, energy, score, acceptance_components,transition = sample_Langevin(x_data, model,L=config['Long_L'],eps=config['eps'],T=T,MH=config["MH"],transition_steps=config['transition_steps'])
        
            utils.plot_multiple_images(x_sample.view(-1,config['im_ch'],config['im_sz'],config['im_sz']),root_path + f"sample_long_{ii}.png")
            utils.plot_transition_images(transition,root_path + f"transition_long_{ii}.png")
            utils.plot_lineplot(acceptance_ratio,root_path + f"acceptance_ratio_long_{ii}.png")
            utils.plot_lineplot(energy,root_path + f"energy_long_{ii}.png")
            utils.plot_lineplot(score,root_path + f"score_long_{ii}.png")
            
            
        
if __name__ == "__main__":

    config = utils.read_json('config.json')
    params_list = ['T',"eps"]
    root_path = config["root_path"]
    for param in params_list:
        root_path = root_path + param + "=" + str(config[param]) + "||"
    root_path = root_path[:len(root_path)-2] + "/"
    
    ## Make Directory if it doesn't Exist 
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    else:
        if config['reset'] == "True":
            utils.delete_contents(root_path)

    utils.copy_folders(['train.py','utils.py','config.json'],root_path)
    
    train(config,root_path,False)
    
    
    
    
    


    



