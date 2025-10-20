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
import torch as t
from PIL import Image
from torch.nn import Softplus
from tqdm import tqdm
import shutil
import os
import json
from model.sampler import sample_HMC, sample_Langevin
from model import models
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
from utils_jgm.tikz_pgf_helpers import tpl_save
import utils
from tqdm import tqdm
import random
import datetime
#from spiral_data import Ring10DDataset
import torch.backends.cuda as bc



def save_results(x_sample, intermediate_results, ii,config,root_path, text=""):



    acceptance_ratio = intermediate_results['acceptance_ratio']
    energy = intermediate_results['energy']
    score = intermediate_results['score']
    utils.plot_multiple_images(x_sample.view(-1, config['im_ch'], config['im_sz'],
                                             config['im_sz']), root_path + f"sample_{text}{ii}.png")
    utils.plot_lineplot(acceptance_ratio, root_path + f"acceptance_ratio_{text}{ii}.png")
    utils.plot_lineplot(energy, root_path + f"energy_{text}{ii}.png")
    utils.plot_lineplot(score, root_path + f"score_{text}{ii}.png")
    return


def train(root_path, resume_checkpoint=False):

    
    start_time = datetime.datetime.now()
    time_to_check = datetime.timedelta(hours=3, minutes=50)
    
    logger = utils.set_logger(root_path+"logging.log")
    config = utils.read_json(root_path+"config.json")
    
    logger.info("STARTED ..........")
    loss_arr = []
    ########################  Get Data ###################################
    if config['data'] == 'flowers':
        utils.download_flowers_data()
    data = {'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
            'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
            'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func)}

    transform = tr.Compose([tr.Resize(config['im_sz']),
                            tr.CenterCrop(config['im_sz']),
                            tr.ToTensor(),
                            tr.Normalize(tuple(0.5*t.ones(config['im_ch'])), 
                                         tuple(0.5*t.ones(config['im_ch'])))])
    # Collect all data and store in q variable
    #q = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform)]).cuda()
    if config['data'] == "lsun_church":
            q = t.load( "/scratch/gilbreth/abdulsal/q_tensor.pt").cuda()
    elif config['data'] == "lsun_bedroom":
            q = t.load( "/scratch/gilbreth/abdulsal/q_tensor2.pt").cuda()
    elif config['data'] == "spiral_data":
            dataset = Ring10DDataset()
            q = dataset.sample(100000).cuda()
    else:    
            q = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform)]).cuda()
   
    # Create bank for persistent chain 
    x_bank = t.randn_like(q)
    x_bank = x_bank.view(-1, config['im_ch']*config['im_sz']*config['im_sz'])

    #################### Intialize model/Optimizer #######################
    
    model = getattr(models, config['model'])(n_c=config['im_ch'],
                                             n_f=config['n_f'],
                                             scaling=config['scaling'])
    model = model.cuda()
    if config['optimizer'] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config['lr_init'])
    elif config['optimizer'] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config['lr_init'])
    eps = config['eps']
    train_iter = 0

    if resume_checkpoint is True:
        checkpoint = utils.find_latest_checkpoint(root_path) # Get path
        checkpoint = torch.load(checkpoint) # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_iter = checkpoint['iteration']
        x_bank = checkpoint['filter_bank']
    
    ############################ Training #################################
    
    logger.info("Training Started ...")
    loss_1_mean = []
    loss_1_std = []
    loss_2_mean = []
    loss_2_std = []
    loss_arr = []
    for ii in range(train_iter, config['num_train_iters']):
    #for ii,data in enumerate(tqdm(train_loader)):
        # Get training data, reshape and add noise
        #logger.info("HERE",ii)
        """
        current_time = datetime.datetime.now()
        time_difference = current_time - start_time
        if time_difference >= time_to_check:
            logger.info("time out")
            break
        """
        x_data  = q[torch.randperm(q.shape[0])[0:config['batch_size']]]
        x_data = x_data.reshape(x_data.shape[0], -1)
        x_data = x_data + config['data_noise'] * torch.randn_like(x_data)
        
        # Intialize starting point of the Langevin chain
        if config['init'] == "noise":
            x_init = torch.randn_like(x_data)
        elif config['init'] == "data":
            x_init = x_data
        elif config['init'] == "persistent":
            x_init, pos = utils.sample_persistent(x_bank, batch_size=config['batch_size'])

        # Set Temperature parameter as required
        if config["T"] == "learnable_T":
            T = model.T
        elif config["T"] == "learnable_log_T":
            T = torch.exp(model.log_T)
        elif not isinstance(config["T"], torch.Tensor):   # If Temperature is not tensor then convert to tensor
            T = t.tensor(config["T"])
            

        # If combined loss run 2 MCMC chains
        if config['combined_loss'] == "True":
            # Get samples init from data or persistent chain 
            if config['sampler'] == "Langevin": 
                if (config["L_data"] == config["L_noise"]) and (config["T_data"] == config["T_noise"]):
    
                    combined_x = torch.cat((x_init, torch.randn_like(x_init)), 0)
                    x_sample_combined, intermediate_results = sample_Langevin(combined_x, model, 
                                                                              dict(config,T=T,L=config['L_data']))
                    x_sample = x_sample_combined[:config['batch_size']]
                    x_sample_noise_init = x_sample_combined[config['batch_size']:]
                    logger.info(f"{x_init.shape}, {combined_x.shape}, {x_sample.shape}, {x_sample_noise_init.shape}")

                    logger.info(f"{x_sample.mean().item()}, {x_sample_noise_init.mean().item()}")
                else:
                    x_sample_noise_init, intermediate_results = sample_Langevin(torch.randn_like(x_init), model, 
                                                                              dict(config,T=config["T_noise"],L=config['L_noise'],eps=1))                    
                    if config["random_length"] == True:
                        L = random.randint(10, 2*config['L_data'])
                        x_sample, intermediate_results = sample_Langevin(x_init, model, 
                                                                    dict(config,T=config["T_data"],L=L))
                    else:
                        logger.info(f"eps = {eps}")
                        x_sample, intermediate_results = sample_Langevin(x_init, model, 
                                                                    dict(config, T=config["T_data"], L=config['L_data'], eps=eps)) 
                        eps = intermediate_results['eps']                                 
            elif config['sampler'] == "HMC":
               
                combined_x = torch.cat((x_init, torch.randn_like(x_init)), 0)
                x_sample_combined, intermediate_results = sample_HMC(combined_x, model, config)
                x_sample = x_sample_combined[:config['batch_size']]
                x_sample_noise_init = x_sample_combined[config['batch_size']:]
            
            #lam = min(25, config['combined_loss_lambda'] + (ii/8000) * (25 - config['combined_loss_lambda']))
            #loss = lam*(model(x_data).mean() - model(x_sample).mean()) + \
            # (model(x_data).mean() - model(x_sample_noise_init).mean()) 
            L_CD = (model(x_data).mean() - model(x_sample).mean()) 
            L_modified_CD = (model(x_data).mean() - model(x_sample_noise_init).mean()) 

            L_CD_std = (model(x_data) - model(x_sample)).std()
            L_modified_std = (model(x_data) - model(x_sample_noise_init)).std()

            loss = config['combined_loss_lambda']*L_CD + L_modified_CD 
            logger.info(f"L_CD = {L_CD}, L_modified_CD = {L_modified_CD}")
            
            loss_1_mean.append(L_CD.detach().cpu().item())
            loss_2_mean.append(L_modified_CD.detach().cpu().item()) 
            loss_1_std.append(L_CD_std.detach().cpu().item())
            loss_2_std.append(L_modified_std.detach().cpu().item())
            
        elif config['multi_noise'] == "True":
            std_devs = np.geomspace(1, 0.01, 10)
            dev = std_devs[random.randint(0,9)]
            x_init = x_data + torch.randn_like(x_init) * dev
            x_sample, intermediate_results = sample_Langevin(x_init, model, dict(config, T=T))
            loss  = model(x_data).mean()  - model(x_sample).mean()
            loss = loss/dev
        elif config['randomized'] == "True":    
            random_draw = np.random.binomial(n=1, p=0.95)
            if random_draw == 1:
                x_sample, intermediate_results = sample_Langevin(x_init, model, dict(config, T=T))
            else:
                x_sample, intermediate_results = sample_Langevin(torch.randn_like(x_init), model, dict(config, T=T))                                              
            
            L = model(x_data) - model(x_sample)
            loss_1_mean.append(L.detach().cpu().mean().item())
            loss_1_std.append(L.detach().cpu().std().item())
            
            loss = model(x_data).mean() - model(x_sample).mean()
             
        else:
            if config['sampler'] == "Langevin":
                logger.info(f"eps = {eps}", "HERE")
                x_sample, intermediate_results = sample_Langevin(x_init, model, dict(config, T=T, eps=eps))
                eps = intermediate_results['eps']


            elif config['sampler'] == "HMC":
                x_sample, intermediate_results = sample_HMC(x_init, model, config)
            
            if config["correction"] == "True":
                correction_factor = intermediate_results["correction_factor"]
                loss = model(x_data) - model(x_sample)
                logger.info(f"{correction_factor.shape}, {loss.shape}, {correction_factor.min()}, {correction_factor.max()}")
                loss = loss * correction_factor   
                loss = loss.mean() 
            else:
                L = model(x_data) - model(x_sample)
                loss_1_mean.append(L.detach().cpu().mean().item())
                loss_1_std.append(L.detach().cpu().std().item())
                          
                loss = model(x_data).mean() - model(x_sample).mean()
                
            if config['learnable_sampler'] == "True":
                loss = loss + 1e-1 * intermediate_results["loss"]
             
        if config['scale_loss'] == "True" or isinstance(config["T"],str) is True:
            loss = loss / T
        
        # If persistent then update the x_bank sample
        if config['init'] == "persistent":
            x_bank[pos] = x_sample.detach().clone()
        # If adaptive eps adjust eps
        if config["adaptive"] == "True":
            """
            eps = utils.adapt(curr_ratio=np.array(intermediate_results['acceptance_ratio']).mean(),
                              threshold=config['adaptive_threshold'],
                              eps=eps)
            """
            score = intermediate_results['score'][0]
            eps =min(1, 0.1/score)
        optimizer.zero_grad()
        loss.backward()
        loss_arr.append(loss.detach().cpu().item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        #loss_arr.append(loss.item())

        for lr_gp in optimizer.param_groups:
            lr_gp['lr'] = max(config['lr_min'], lr_gp['lr'] * config['lr_decay'])

        logger.info(f"{ii},loss = {loss.sum().item()},Temperature = {T.item()}")
        logger.info(f"shortRun chain ---> maxVal={x_sample.max().item()}, minVal = {x_sample.min().item()}")
        #logger.info(f"eps = {eps}")
        logger.info(f"acceptance = {intermediate_results['acceptance_ratio'][-1]}")
        logger.info(f"score_start = {intermediate_results['score'][0]}, score_end = {intermediate_results['score'][-1]}")
        #print(ii,loss.item(),T.item())
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        
        if ii % config["logging_freq"] == 0:
            
            save_results(x_sample, intermediate_results, ii, config, root_path)
            utils.plot_multiple_images(x_init.view(-1, config['im_ch'], config['im_sz'],
                                                config['im_sz']), root_path + f"sampleInit_{ii}.png")
            if config['combined_loss'] == "False":
                x_sample_noise_init, intermediate_results = sample_Langevin(torch.randn_like(x_init), model, 
                                                                                dict(config,T=config["T_noise"],
                                                                                    L=config['L_noise'],eps=1))
            
            utils.plot_multiple_images(x_sample_noise_init.view(-1, config['im_ch'], config['im_sz'],
                                                config['im_sz']), root_path + f"sampleNoiseInit_{ii}.png")
            
            
            """
            x_sample_noise_init, intermediate_results_noise = sample_Langevin(torch.randn(10000,10).cuda(), 
                                                                                  model, dict(config, 
                                                                                              L=50,
                                                                                              T=5e-5,
                                                                                              eps=5e-1)) 
            dataset.plot_model_density(model, T, save_path=root_path + f"model_density_{ii}.png")
            dataset.plot_samples_plane(x_data, save_path=root_path + f"true_data_KDE_{ii}.png")
            dataset.plot_samples_plane(x_sample_noise_init.view(10000,10,1,1), save_path=root_path + f"model_KDE_{ii}.png")
            """
            torch.save({
              'iteration': ii,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'filter_bank': x_bank
              }, root_path + f'Tempstate-{ii}.pth')
            
            if os.path.exists(root_path + f'Tempstate-{ii}.pth'):
            # Rename the temporary file to the final filename
                os.rename(root_path + f'Tempstate-{ii}.pth', root_path + f'state-{ii}.pth')
            """
            torch.save({
              'iteration': ii,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, root_path + f'/state.pth')
            """
            #np.save(root_path + f"loss_1_{ii}.npy", loss_arr)
    
            np.save(root_path + f"loss_{ii}.npy", np.array(loss_arr))
            np.save(root_path + f"loss_1_mean_{ii}.npy", np.array(loss_1_mean))
            np.save(root_path + f"loss_2_mean_{ii}.npy", np.array(loss_2_mean))
            np.save(root_path + f"loss_1_std_{ii}.npy", np.array(loss_1_std))
            np.save(root_path + f"loss_2_std_{ii}.npy", np.array(loss_2_std))
     
        if (ii+1)%config["long_run_freq"]==0:
            
            if config['sampler'] == "Langevin":
                x_sample_data_init, intermediate_results_data = sample_Langevin(x_data, model,
                                                                                dict(config, L=config['Long_L'],T=T))            
                x_sample_noise_init, intermediate_results_noise = sample_Langevin(torch.randn_like(x_data), 
                                                                                  model, dict(config, L=config["Long_L"],T=T))                                      
            elif config['sampler'] == "HMC":
                x_sample_data_init, intermediate_results_data = sample_HMC(x_data, model,
                                                                            dict(config, HMC_steps=config['HMC_steps_Long']))
                x_sample_noise_init, intermediate_results_noise = sample_HMC(torch.randn_like(x_data), model,
                                                                              dict(config, HMC_steps=config['HMC_steps_Long']))
            
            logger.info(f"longRun chain ---> maxVal={x_sample.max().item()}, \
                           minVal = {x_sample.min().item()}")
            
            save_results(x_sample_data_init, intermediate_results_data, ii, config,root_path, text="long_DataInit_")
            save_results(x_sample_noise_init, intermediate_results_noise, ii, config,root_path, text="long_NoiseInit_")
            np.save(root_path + f"energyArrayDataLong_{ii}.npy", intermediate_results_data['energy'])
            np.save(root_path + f"energyArrayNoiseLong_{ii}.npy", intermediate_results_noise['energy'])
            """
            utils.tensors_to_gif(intermediate_results_data['transition'], int(config['batch_size']**0.5),
                                 gif_filename=root_path + f"transition_long_DataInit_{ii}.gif")
            utils.tensors_to_gif(intermediate_results_noise['transition'], int(config['batch_size']**0.5),
                                 gif_filename=root_path + f"transition_long_NoiseInit_{ii}.gif")
            """
            print(len(intermediate_results_data['transition']))
           
if __name__ == "__main__":

    config = utils.read_json('config.json')
    params_list = ['T',"eps","L","MH","combined_loss_lambda"]
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
    
    train(root_path,False)
