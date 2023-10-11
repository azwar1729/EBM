from torchvision import utils
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import math
import logging
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # NOQA
from PIL import Image  # NOQA
import shutil
import os
import torchvision.utils as vutils
import torchvision as tv

def set_logger(filename):
    
    logging.basicConfig(filename=filename,
                    format='%(asctime)s %(message)s',
                    filemode='a')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    return logger


def read_json(fname):
    with open(fname, 'r') as file:
        data = json.load(file)
    return data

def copy_folders(src_list, dest_dir):
    """ Copy a list of directory and files to a given destination file """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for src in src_list:
        if os.path.exists(src):
            # Copy the entire directory
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(dest_dir, os.path.basename(src)))
            # Copy individual files
            elif os.path.isfile(src):
                shutil.copy2(src, dest_dir)
        

def delete_contents(directory_path):
    """Deletes everything inside a directory."""
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def plot_single_images(image,path):
    print(image.shape)
    plt.imshow(image.detach().cpu().numpy().transpose(1,2,0))
    plt.savefig(path)
    plt.close()

def plot_multiple_images(image_tensor, path): tv.utils.save_image(torch.clamp(image_tensor, -1., 1.), path, normalize=True, nrow=int(image_tensor.shape[0] ** 0.5))

def plot_transition_images(image_list,path):
    
    tensor_images = torch.stack(image_list)
    grid = vutils.make_grid(tensor_images, nrow=len(image_list))
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    grid_np = grid.detach().cpu().numpy().transpose((1, 2, 0))
    plt.imshow(grid_np)
    plt.savefig(path)
    plt.close()

def plot_lineplot(input_list,path):
    
    plt.figure()
    plt.plot(input_list)
    plt.savefig(path)
    plt.close()

