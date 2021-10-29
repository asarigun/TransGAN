# TransGAN: Two Transformers Can Make One Strong GAN [[YouTube Video]](https://www.youtube.com/watch?v=xwrUkHiDoiY)

Paper Authors: Yifan Jiang, Shiyu Chang, Zhangyang Wang

*CVPR 2021*

<!--
<p align="center">
  <a href="https://arxiv.org/abs/2102.07074"><img src="https://img.shields.io/badge/Paper-Report-red"/></a>
  <a href="https://www.youtube.com/watch?v=xwrUkHiDoiY"><img src="https://img.shields.io/badge/YouTube-Video-ff69b4"/></a>
  <a href="https://github.com/asarigun/TransGAN/blob/main/LICENSE"><img src="https://img.shields.io/github/license/thudm/cogdl"/></a>
</p>
-->

This is re-implementation of [TransGAN: Two Transformers Can Make One Strong GAN, and That Can Scale Up, CVPR 2021](https://arxiv.org/abs/2102.07074) in PyTorch.

<p align="center"><img width="100%" src="https://github.com/asarigun/TransGAN/blob/main/images/transgan.jpg"></p>

Generative Adversarial Networks-GAN builded completely free of Convolutions and used Transformers architectures which became popular since Vision Transformers-[ViT](https://arxiv.org/abs/2010.11929). In this implementation, CIFAR-10 dataset was used. 

<!--
<p float="center">
  <img src="https://github.com/asarigun/TransGAN/blob/main/results/transgan_mnist1.gif" width="49%" />
  <img src="https://github.com/asarigun/TransGAN/blob/main/results/transgan_mnist1.gif" width="49%" />
</p> -->
<!--
<p align="center"><img width="30%" src="https://github.com/asarigun/TransGAN/blob/main/results/transgan_mnist1.gif"></p>
<p align="center"><img width="30%" src="https://github.com/asarigun/TransGAN/blob/main/images/atransgan_cifar.gif"></p>-->

<table>
<tr>
<td style="text-align: center">0 Epoch</td>
<td style="text-align: center">40 Epoch</td> 
<td style="text-align: center">100 Epoch</td>
<td style="text-align: center">200 Epoch</td> 
</tr>
<trt>
<p align="center"><img width="30%" src="https://raw.githubusercontent.com/asarigun/TransGAN/main/images/atransgan_cifar.gif"></p>
</tr>
<tr>
<td> <img src="https://raw.githubusercontent.com/asarigun/TransGAN/main/results/0.jpg" style="width: 400px;"/> </td>
<td> <img src="https://raw.githubusercontent.com/asarigun/TransGAN/main/results/40.jpg" style="width: 400px;"/> </td>
<td> <img src="https://raw.githubusercontent.com/asarigun/TransGAN/main/results/100.jpg" style="width: 400px;"/> </td>
<td> <img src="https://raw.githubusercontent.com/asarigun/TransGAN/main/results/200.jpg" style="width: 400px;"/> </td>
</tr>
</table>


## Related Work - Vision Transformers ([ViT](https://openreview.net/pdf?id=YicbFdNTTy))

In this implementation, as a discriminator, Vision Transformer(ViT) Block was used. In order to get more info about ViT, you can look at the original paper [here](https://arxiv.org/abs/2010.11929)

<p align="center"><img width="80%" src="https://github.com/asarigun/TransGAN/blob/main/images/vit.gif"></p>

Credits for illustration of ViT: [@lucidrains](https://github.com/lucidrains) 

## Installation

Before running ```train.py```, check whether you have libraries in ```requirements.txt```! Also, create ```./fid_stat``` folder and download the [fid_stats_cifar10_train.npz](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz) file in this folder. To save your model during training, create ```./checkpoint``` folder using ```mkdir checkpoint```.

## Training 

```bash
python train.py
```
## Pretrained Model

You can find pretrained model [here](https://drive.google.com/file/d/134GJRMxXFEaZA0dF-aPpDS84YjjeXPdE/view). You can download using:

```bash

wget https://drive.google.com/file/d/134GJRMxXFEaZA0dF-aPpDS84YjjeXPdE/view

```
or 

```bash

curl gdrive.sh | bash -s https://drive.google.com/file/d/134GJRMxXFEaZA0dF-aPpDS84YjjeXPdE/view

```

## License

MIT

## Citation
```bibtex
@article{jiang2021transgan,
  title={TransGAN: Two Transformers Can Make One Strong GAN},
  author={Jiang, Yifan and Chang, Shiyu and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2102.07074},
  year={2021}
}
```
```bibtex
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
```bibtex
@inproceedings{zhao2020diffaugment,
  title={Differentiable Augmentation for Data-Efficient GAN Training},
  author={Zhao, Shengyu and Liu, Zhijian and Lin, Ji and Zhu, Jun-Yan and Han, Song},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
