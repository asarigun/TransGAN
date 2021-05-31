# TransGAN-S for CIFAR-10

Here in this part of repository, it has been tried to reproduce the TransGAN-S {5,2,2} for CIFAR-10 dataset. However, when it was trained on some given hyperparameters, it was not able to reproduced which is far away from given results; aimed FID score 41.41 but got 252 which is terribly far away from the goal. 

However, the model was saved in order to share people who curious to try and want to optimize it.

## So let's start training stage!

Before training,  please download  [fid_stats_cifar10_train.npz](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz) to ```./fid_stat```.

Then after getting fid_stat, change your tool CPU to CUDA. Although this model relatively small model in the paper, it may take hours (for 200 epochs, it takes nearly 23 hours) or maybe days to train it. 

## Hyperparameters for CIFAR-10

This is the hyperparameters that has been used when training on CIFAR-10 dataset.

### Training details reported by paper authors

```
lr_gen = 0.0001 ---> Learning rate for generator
lr_dis = 0.0001 ---> Learning rate for discriminator
latent_dim = 384 ---> Latent dimension
gener_batch_size = 60 ---> Batch size for generator
dis_batch_size = 30 ---> Batch size for discriminator
```

### Training hyperparameters given by code author

```
epoch = 200 ---> Number of epoch
weight_decay = 1e-3 ---> Weight decay
drop_rate = 0.5 ---> dropout
```

### Architecture details

```
image_size = 32 ---> H,W size of image for discriminator
initial_size = 8 ---> Initial size for generator
patch_size = 16 ---> Patch size for generated image
num_classes = 1 ---> Number of classes for discriminator 
output_dir = 'checkpoint' ---> saved model path
dim = 384 ---> Embedding dimension 
depth = 5 ---> depth for transformers encoder block for discriminator
depth1 = 5 ---> depth for first transformers encoder block-set which is after MLP block for generator
depth2 = 2 ---> depth for second transformers encoder block-set which is after first encoder block for generator
depth3 = 2 ---> depth for third transformers encoder block-set which is after second encoder block for generator
heads = 8 ---> head for attention mechanism
```

## Pre-trained Model

As it was stated above, the pre-trained model was not good-terribly bad! However, it was good idea to share with you, you can try it and see how TransGAN so heavy. You can reach the saved models from the [link](https://drive.google.com/drive/folders/1CguFSf4FGiubQAMZIZSm7o9-a52QP0ht?usp=sharing). If you have good idea to train it faster and get best metric results, please do not hesitate to contact!



