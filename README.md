# Context-Encoder
I modify this network to be used for image inpainting.And it can be applied to inpaint most missing regions.

This is a part of our paper which is published on the Applied Optics Journal. （[https://www.osapublishing.org/ao/fulltext.cfm?uri=ao-60-26-8198](https://www.osapublishing.org/ao/fulltext.cfm?uri=ao-60-26-8198))

Recently,this paper has been selected as Spotlight.([https://www.osapublishing.org/spotlight/summary.cfm?id=458598](https://www.osapublishing.org/spotlight/summary.cfm?id=458598))

It is my honor to have the first paper of my BS learning career.

Hope you can star this repository or even cite our paer and sincerely hope that it can help you!

## Introduction
This project, which I wrote and improved Context Encoder, selected the currently hot pytorch deep learning framework developed by Facebook, and re-wrote and produced datasets by myself, as explained below.

## Environment
Anaconda (includes pytorch1.7.1, numpy, matplotlib, PIL, opencv-python, etc.)

python3.8.5
## Software architecture
Now I'll introduce our engineering source code to the ".py" files
### cfg.py 
This file is a configuration file for configuring data path for training set and some parameters of the training process, such as batchsize and epoch_number

Note: Context Encoder is unsupervised learning and does not require mannual labels, as baseline, we still introduce labels which is the same as the datas. In fact they are not necessary. We just introduce them to save the raw data because the original data will be processed before inputing.

### dataset.py 
This file is the dataset used by the user to rewrite the pytorch dataset.

We overwrite __ init __(),__ getitem __ () method and __ len __ () method.

You can run this program to see our processes on the raw image. This is the core part of our unsupervised learning.

### netG.py 
This file writes the structure and forward propagation of the generator Generator

### netD.py 
This file writes the structure and forward propagation of the judge Discriminator

### new_train.py 
This file is a program to train the generative network and the discriminative network. Before training, you need to write the cfg.py file to determine your hyper parameters.

### predict.py
You can use your trained model to do the prediction. Using this, you can actually do an image inpainting work.

Actually, you must know that when you run this program, you need to input an image and a mask. Or you can input an incomplete image whose missing region grayscale is 255.

## "dataset" document
The dataset document contains "train" document and "train_label" document. As I discribe above, these two are same. We introduct "train_label" document to save the raw data, and the images in "train" document will be processed when constructing our dataset and dataloader.


