from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from tqdm import tqdm


from utils import *
from models import *
from fid_score import *
from inception_score import *


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--image_size', type=int, default= 28 , help='Size of image for discriminator input.')
parser.add_argument('--initial_size', type=int, default=8 , help='Initial size for generator.')
parser.add_argument('--patch_size', type=int, default=16 , help='Patch size for generated image.')
parser.add_argument('--num_classes', type=int, default=1 , help='Number of classes for discriminator.')
parser.add_argument('--lr_gen', type=int, default=0.0001 , help='Learning rate for generator.')
parser.add_argument('--lr_dis', type=int, default=0.0001 , help='Learning rate for discriminator.')
parser.add_argument('--weight_decay', type=int, default=1e-3 , help='Weight decay.')
parser.add_argument('--latent_dim', type=int, default=128 , help='Latent dimension.')
#parser.add_argument('--n_critic', type=int, default=5 , help='n_critic.')
parser.add_argument('--gener_batch_size', type=int, default=60 , help='Batch size for generator.')
parser.add_argument('--dis_batch_size', type=int, default=30 , help='Batch size for discriminator.')
parser.add_argument('--epoch', type=int, default=200 , help='Number of epoch.')
parser.add_argument('--output_dir', type=str, default='checkpoint' , help='Checkpoint.')
parser.add_argument('--dim', type=int, default=128 , help='Embedding dimension.')
parser.add_argument('--img_name', type=str, default="img_name" , help='Name of pictures file.')


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
print("Device:",device)


generator= Generator(depth1=1, depth2=1, depth3=1, initial_size=7, dim=128, heads=1, mlp_ratio=4, drop_rate=0.5)

discriminator = Discriminator(image_size=28, patch_size=14, input_channel=1, num_classes=1, dim=128, depth=1, heads=1, mlp_ratio=4, drop_rate=0.5)
discriminator.to(device)


generator.apply(inits_weight)
discriminator.apply(inits_weight)

optim_gen = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()),
                lr=0.001, weight_decay=1e-4)

optim_dis = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                lr=0.001, weight_decay=1e-4)

fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'

writer=SummaryWriter()
writer_dict = {'writer':writer}
writer_dict["train_global_steps"]=0
writer_dict["valid_global_steps"]=0


def train(noise,generator, discriminator, optim_gen, optim_dis,
        epoch, writer,img_size=28, latent_dim = 384,
        gener_batch_size=60,device="cpu"):


    writer = writer_dict['writer']
    gen_step = 0

    generator = generator.train()
    discriminator = discriminator.train()


    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=30, shuffle=True)

    for index, (img, _) in enumerate(train_loader):

        global_steps = writer_dict['train_global_steps']

        #real_imgs = img.type(torch.cuda.FloatTensor)
        real_imgs = img.type(torch.FloatTensor)

        #noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (img.shape[0], latent_dim)))
        
        noise = torch.FloatTensor(np.random.normal(0, 1, (img.shape[0], latent_dim)))

        optim_dis.zero_grad()
        real_valid=discriminator(real_imgs)
        fake_imgs = generator(noise).detach()

        assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"

        fake_valid = discriminator(fake_imgs)

        loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).to(device) + torch.mean(nn.ReLU(inplace=True)(1 + fake_valid)).to(device)

        loss_dis.backward()
        optim_dis.step()

        writer.add_scalar("loss_dis", loss_dis.item(), global_steps)

        if global_steps % 5 == 0:

            optim_gen.zero_grad()

            gener_noise = torch.FloatTensor(np.random.normal(0, 1, (gener_batch_size, latent_dim)))

            generated_imgs= generator(gener_noise)
            fake_valid = discriminator(generated_imgs)

            gener_loss = -torch.mean(fake_valid).to(device)
            gener_loss.backward()
            optim_gen.step()
            writer.add_scalar("gener_loss", gener_loss.item(), global_steps)

            gen_step += 1


        if gen_step and index % 500 == 0:
            sample_imgs = generated_imgs[:25]
            img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)           
            tqdm.write("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch+1, index % len(train_loader)+50, len(train_loader), loss_dis.item(), gener_loss.item()))



def validate(generator, writer_dict, fid_stat):



        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']

        generator = generator.eval()

        writer_dict['valid_global_steps'] = global_steps + 1




for epoch in range(epoch):

    train(noise, generator, discriminator, optim_gen, optim_dis,
    epoch, writer,img_size=28, latent_dim = 128,
    gener_batch_size=60)

    checkpoint = {'epoch':epoch, 'best_fid':best}
    checkpoint['generator_state_dict'] = generator.state_dict()
    checkpoint['discriminator_state_dict'] = discriminator.state_dict()

    score = validate(generator, writer_dict, fid_stat)



checkpoint = {'epoch':epoch, 'best_fid':best}
checkpoint['generator_state_dict'] = generator.state_dict()
checkpoint['discriminator_state_dict'] = discriminator.state_dict()
score = validate(generator, writer_dict, fid_stat) 
save_checkpoint(checkpoint, output_dir=output_dir)

