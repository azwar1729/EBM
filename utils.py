from torchvision import utils
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
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
import imageio
def set_logger(filename):
    
    logging.basicConfig(filename=filename,
                    format='%(asctime)s %(message)s',
                    filemode='a')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    return logger


def read_json(path):
    try:
        with open(path, 'r') as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError:
        return {}

def write_json(data_dict,path):
    
    with open(path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)    
    return

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


def tensor_to_image(tensor, n):
    # Convert tensor to a grid of images
    grid = vutils.make_grid(torch.clamp(tensor, -1, 1.), normalize=True, nrow=int(n))
    # Convert grid to numpy array
    grid_np = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return grid_np


def tensors_to_gif(tensor_list, n, gif_filename="transition.gif"):
    images = []
    for tensor in tensor_list:
        img_np = tensor_to_image(tensor, n)
        images.append(img_np)

    imageio.mimsave(gif_filename, images, duration=1.5,loop=0)         


def plot_lineplot(input_list,path):
    plt.figure()
    plt.plot(input_list)
    plt.savefig(path)
    plt.close()


def sample_persistent(state_set, batch_size=64):
    rand_inds = torch.randperm(state_set.shape[0])[0:batch_size]
    return state_set[rand_inds], rand_inds

def download_flowers_data():
    import tarfile
    try:
        from urllib.request import urlretrieve
    except ImportError:
        from urllib import urlretrieve

    dataset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/flowers/')
    if not os.path.exists(os.path.join(dataset_folder, "jpg")):
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        print('Downloading data from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/ ...')
        tar_filename = os.path.join(dataset_folder, "102flowers.tgz")
        if not os.path.exists(tar_filename):
            urlretrieve("http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", tar_filename)

        # extract flower images from tar file
        print('Extracting ' + tar_filename + '...')
        tarfile.open(tar_filename).extractall(path=dataset_folder)

        # clean up
        os.remove(tar_filename)
        print('Done.')
    else:
        print('Data available at ' + dataset_folder)
