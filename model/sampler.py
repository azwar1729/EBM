import torch as t
import torch
import numpy as np
import logging


def leapfrog(model, q, p, Leapfrog_steps, eps):  
    # Half step of momentum #
    grad = t.autograd.grad(model(q).sum(), [q])[0]
    p = p - eps*grad/2
    # LeapFrog steps #
    for i in range(1, Leapfrog_steps+1):
        q = q + eps*p
        if i != Leapfrog_steps:
            grad = t.autograd.grad(model(q).sum(), [q])[0]
            p = p - eps*grad

    # Complete half step #
    grad = t.autograd.grad(model(q).sum(), [q])[0]
    p = p - eps*grad/2

    return q, p, grad


def HMC_acceptance(model, curr_q, curr_p, proposed_q, proposed_p, mass=1):    
    batch_size = curr_q.shape[0]
    curr_U = model(curr_q)
    proposed_U = model(proposed_q)
    curr_K = ((curr_p**2).view(batch_size, -1)).sum(axis=1)
    proposed_K = ((proposed_p**2).view(batch_size, -1)).sum(axis=1)

    ratio = t.exp(0.5*(curr_K - proposed_K)/(mass**2)
                        + curr_U - proposed_U).clamp(0, 1).mean().item()

    return ratio

def adapt(curr_ratio, threshold, eps):

    diff = (curr_ratio - threshold)
    diff = diff/10
    eps = eps * (1 + diff)
    return eps


def sample_HMC(q, model, config):

    Leapfrog_steps = config["Leapfrog_steps"]
    HMC_steps = config['HMC_steps']
    eps = config["eps"]
    gamma = config['gamma']
    mass = config['mass']
    metropolis = config["MH"]
    adaptive = config['adaptive']
    adaptive_threshold = config['adaptive_threshold']
    transition_steps = config['transition_steps']
    
    acceptance_ratio = []
    energy = []
    score = []
    transition = []
    # Get current q and p ( p draw from random dist)
    curr_q = q.clone()
    curr_p = t.randn_like(q) * mass
    # Set gradient for curr_q as true to calc gradient for leapfrog
    curr_q.requires_grad = True

    for i in range(HMC_steps):
        accept = 0
        while accept == 0:          
            # Run Leapfrog
            proposed_q, proposed_p, grad = leapfrog(model, curr_q, curr_p,
                                                     Leapfrog_steps, eps)
            # Get Acceptance ratio
            ratio = HMC_acceptance(model, curr_q, curr_p,
                                   proposed_q, proposed_p, mass)
            acceptance_ratio.append(ratio)         
            # pick binomial R.V based on acceptance ratio  
            accept = np.random.binomial(n=1, p=ratio)
            # If adaptive adjustment for eps set to True
            if adaptive == "True":
                eps = adapt(curr_ratio=accept,
                            threshold=adaptive_threshold,
                            eps=eps)
               
            # if metroplis is "False" run unadjusted HMC
            if metropolis == "False":
                accept = 1
            if accept == 1:
                # Update q and p 
                curr_q = proposed_q
                curr_p = proposed_p*gamma \
                    + ((1 - gamma**2)**0.5) * t.randn_like(curr_p) * mass
                
                energy.append(model(curr_q).mean().item())
                score.append((grad**2).mean().item())

                if i % transition_steps == 0:
                    transition.append(curr_q.view(-1, config["im_ch"], 32, 32))
                
    #return curr_q.detach(), score, acceptance_ratio, energy
    print(eps)
    intermediate_results = {"acceptance_ratio": acceptance_ratio,
                            "energy": energy,
                            "score": score,
                            "transition": transition,
                            "eps": eps}
    
    return curr_q.detach(), intermediate_results

   
def langevin_acceptance(x, x_star, grad, grad_star, model, eps, T):
    
    term1 = t.norm(x_star - x + eps * grad, dim=1) ** 2   
    term2 = t.norm(x - x_star + eps * grad_star, dim=1) ** 2
    
    energy_diff = model(x)/T - model(x_star)/T

    ratio = (1 / (4 * eps * T)) * (term1 - term2) + energy_diff
    ratio = ratio.clamp(-1e2, 1e2)
    ratio = t.exp(ratio).clamp(0, 1).mean().item()
    
    return ratio 
    

def sample_Langevin(x, model, config):
    
    # Set all hyperparams from config 
    L = config["L"]
    eps = config['eps']
    T = config['T']
    MH = config['MH']
    adaptive = config['adaptive']
    adaptive_threshold = config['adaptive_threshold']
    transition_steps = config['transition_steps']
    ch = config['im_ch']

    # Lists to store intermediate values
    acceptance_ratio = []  # List to track acceptance ratio across langevin iterations
    energy = []  # List to keep track of energy across langevin iterations
    score = [] # List to keep track of score magnitude across langevin iterations
    transition = [] # List to keep of image transitions

    loss = t.tensor(0) #OPTIONAL USAGE Loss keeps track of loss involed with LD 

    for i in range(L):

        x.requires_grad = True  # set gradient flag of x is true for Langevin 
        grad = torch.autograd.grad(model(x).sum(), [x], create_graph=True)[0]
        accept = 0

        while accept == 0:
            
            x_star = x - eps*grad + t.sqrt(2 * eps * T)*t.randn_like(x)
            grad_star = torch.autograd.grad(model(x_star).sum(), [x_star], create_graph=True)[0]

            if MH == "False":
                accept = 1
                ratio = 1
            else:
                ratio = langevin_acceptance(x, x_star, grad, grad_star, model, eps, T)
                accept = np.random.binomial(n=1, p=ratio)            
                if adaptive == "True":
                    eps = adapt(curr_ratio=accept,
                                threshold=adaptive_threshold,
                                eps=eps)
            
            acceptance_ratio.append(ratio)
        print(ratio)
        lambd = 5
        loss = loss +  lambd**2/(ratio * ((x_star - x) ** 2).sum()) - ratio/lambd**2 * ((x_star - x) ** 2).sum()
        #print(loss)
        x = x_star.detach().clone()
        x = x.detach().clone()
    
        energy.append(model(x).mean().item())
        score.append((grad**2).sum(axis=1).mean().item())  

        if i % transition_steps == 0:
            transition.append(x.view(-1, ch, 32, 32))

    intermediate_results = {"acceptance_ratio": acceptance_ratio,
                            "energy": energy,
                            "score": score,
                            "transition": transition,
                            "loss": loss}
    
    return x.detach(), intermediate_results
    


