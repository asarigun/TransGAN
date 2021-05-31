# TransGAN 

This is re-implementation of [TransGAN](https://arxiv.org/abs/2102.07074) in PyTorch.

<p align="center"><img width="100%" src="https://github.com/asarigun/TransGAN/blob/main/images/transgan.jpg"></p>

Generative Adversarial Networks-GAN builded completely free of Convolutions and used Transformers architectures which became popular since Vision Transformers-[ViT](https://arxiv.org/abs/2010.11929). In this implementatio, MNIST and CIFAR-10 dataset were used. For MNIST, you can look at [here](https://github.com/asarigun/TransGAN/blob/main/main.ipynb).

<p float="center">
  <img src="https://github.com/asarigun/TransGAN/blob/main/results/transgan_mnist1.gif" width="49%" />
  <img src="https://github.com/asarigun/TransGAN/blob/main/results/transgan_mnist1.gif" width="49%" />
</p>
<!--
<p align="center"><img width="30%" src="https://github.com/asarigun/TransGAN/blob/main/results/transgan_mnist1.gif"></p>
<p align="center"><img width="30%" src="https://github.com/asarigun/TransGAN/blob/main/results/transgan_mnist2.gif"></p>-->

For CIFAR-10 dataset you can look at detailed README [here](https://github.com/asarigun/TransGAN/blob/main/cifar/README.md) at ```./cifar``` directory.

## Related Work - Vision Transformers ([ViT](https://openreview.net/pdf?id=YicbFdNTTy))

In this implementation, as a Transformer Block, Vision Transformer(ViT) Block was used. In order to get more info about ViT, you can look at the original paper [here](https://arxiv.org/abs/2010.11929)

<p align="center"><img width="80%" src="https://github.com/asarigun/TransGAN/blob/main/images/vit.gif"></p>

Credits for illustration of ViT: @lucidrains 


## Citation
```
@article{jiang2021transgan,
  title={TransGAN: Two Transformers Can Make One Strong GAN},
  author={Jiang, Yifan and Chang, Shiyu and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2102.07074},
  year={2021}
}
```
```
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
