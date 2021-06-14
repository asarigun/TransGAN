import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

import os

import numpy as np


def noise(n_samples, z_dim, device):
        return torch.randn(n_samples,z_dim).to(device)

def inits_weight(m):
        if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight.data, 1.)


def noise(imgs, latent_dim):
        return torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))

def gener_noise(gener_batch_size, latent_dim):
        return torch.FloatTensor(np.random.normal(0, 1, (gener_batch_size, latent_dim)))

def save_checkpoint(states,is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))
