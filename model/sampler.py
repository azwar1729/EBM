import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision as tv
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
import torch
import numpy as np
import shutil
import os
import argparse
import sys
import json

def leapfrog(model,q,p,Leapfrog_steps,eps):
    
    #### Half step of momentum #####
    grad = t.autograd.grad(model(q).sum(), [q])[0]
    curr_grad = grad.clone()
    p = p - eps*grad/2
    
    ## LeapFrog steps ####
    for i in range(1,Leapfrog_steps+1):
        
        q = q + eps*p
        
        if i!=Leapfrog_steps:
            grad = t.autograd.grad(model(q).sum(), [q])[0]
            p = p - eps*grad

    ##### Complete half step ########
    grad = t.autograd.grad(model(q).sum(), [q])[0]
    p = p - eps*grad/2

    return q,p


def sample_HMC(q, model,Leapfrog_steps=3,HMC_steps=100,eps=0.01,gamma=0.9,mass=1e-2,metropolis=True):

    batch_size = q.shape[0]
    acceptance_ratio = []
    energy = []
    score = []
    
    curr_q = q.clone()
    
    curr_p = t.randn_like(q) *mass
    curr_q.requires_grad = True

    for k in range(HMC_steps):  
        
        accept = 0
        while accept == 0:
            
            
            proposed_q,proposed_p = leapfrog(model,curr_q,curr_p,Leapfrog_steps,eps)

            #### Check ratio ######
            curr_U = model(curr_q)
            proposed_U = model(proposed_q)

            curr_K = ((curr_p**2).view(batch_size,-1)).sum(axis=1)
            proposed_K = ((proposed_p**2).view(batch_size,-1)).sum(axis=1)

            ratio = t.exp(  0.5*(curr_K - proposed_K )/(mass**2) + curr_U - proposed_U  ).clamp(0,1).mean().item()
            acceptance_ratio.append(ratio)
            energy.append(proposed_U.mean().item())
            #score.append((grad**2).sum(axis=1).mean().item())
            acceptance_ratio.append(ratio)
            
            accept =  np.random.binomial(n=1, p=ratio)

            if metropolis == False:
                accept = 1
            if accept == 1:
 
                curr_q = proposed_q
                curr_p = proposed_p*gamma + ((1 - gamma**2)**0.5) * t.randn_like(curr_p) * mass
                
    return curr_q.detach(),acceptance_ratio, energy, score



def sample_Langevin(x, model,L=150,eps=1,T=5e-5,MH=False,transition_steps=100):
    """
    L --> Total number of Langevin Iterations
    eps --> step size
    T --> Temperature
    MH --> if True carry out Metropolis adjust step
    transition_steps --> eg:100, every 100 steps images are saved to see how transition looks like in the langevin chain
    """
    acceptance_ratio = []  # List to track acceptance ratio across langevin iterations
    energy = []  # List to keep track of energy across langevin iterations
    score = [] # List to keep track of score magnitude across langevin iterations
    transition = [] # List to keep of image transitions
    
    acceptance_energy_term = []  # Diff of Energy in acceptance term
    acceptance_score_term = []   # Diff in L2 terms
    
    x.requires_grad = True  # set gradient flag of x is true for Langevin 
    
    for i in range(L):

        grad = torch.autograd.grad(model(x).sum(), [x])[0]
        accept = 0
        while accept == 0:
            
            x_star = x - eps*grad + t.sqrt(2 * eps * T)*t.randn_like(x)
            grad_star = torch.autograd.grad(model(x_star).sum(), [x_star])[0]

            term1 = t.norm(x_star - x + eps * grad, dim=1) ** 2    
            term2 = t.norm(x - x_star + eps * grad_star, dim=1) ** 2
            
            energy_diff = model(x)/T - model(x_star)/T

            ratio =  (1 / (4 * eps * T)) * (term1 - term2) + energy_diff
            ratio = ratio.clamp(-1e2,1e2)
            ratio = t.exp(ratio).clamp(0,1).mean().item()
            accept = np.random.binomial(n=1,p=ratio)
            
            if MH=="False":
                accept = 1
            
            acceptance_energy_term.append(energy_diff.mean().item())
            acceptance_score_term.append( ((1 / (4 * eps * T)) * (term1 - term2)).mean().item() )
            acceptance_ratio.append(ratio)
            energy.append(model(x).mean().item())
            score.append((grad**2).sum(axis=1).mean().item())
    
        x = x_star 
        if i%transition_steps==0:
            transition.append(x.view(-1,1,32,32)[0])
    
    acceptance_components = np.array([acceptance_energy_term, acceptance_score_term])
    return x.detach(),acceptance_ratio, energy, score, acceptance_components,transition
    

