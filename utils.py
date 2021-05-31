import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

import os

import numpy as np



def train_loader(batch_size=30, img_size=28):

    transform = transforms.Compose([transforms.Resize(size=(28, 28)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=30, shuffle=True)

    return train_loader 

def valid_loader(batch_size=30, img_size=28):

    transform = transforms.Compose([transforms.Resize(size=(28, 28)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=30, shuffle=False)
    valid_loader = test_loader

    return valid_loader


def inits_weight(m):
        if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight.data, 1.)


def gener_noise(gener_batch_size, latent_dim):
        return torch.FloatTensor(np.random.normal(0, 1, (gener_batch_size, latent_dim)))

def save_checkpoint(states, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))

train_loader(batch_size=30, img_size=28)
valid_loader(batch_size=30, img_size=28)
