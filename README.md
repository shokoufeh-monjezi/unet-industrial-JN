# Getting Start with U-Net Industrial

## Image Segmentation
Image segmentation is the process of partitioning a digital image into multiple segments. The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze. The result of image segmentation is a set of segments that collectively cover the entire image, or a set of contours extracted from the image.
##  Model Overview
This U-Net model is adapted from the original version of the U-Net model <https://arxiv.org/abs/1505.04597> which is a convolutional auto-encoder for 2D image segmentation.

This work proposes a modified version of U-Net, called TinyUNet which performs efficiently and with very high accuracy on the industrial anomaly dataset DAGM2007. TinyUNet, like the original U-Net is composed of two parts:

*An encoding sub-network*
*A decoding sub-network*:
What is encoder decoder in deep learning models? The Encoder converts the input sequence into a single dimensional vector. The decoder uses the output of the encoder as the input and it converts this vector into the output sequence.

This model repeatedly applies 3 downsampling blocks composed of two 2D convolutions followed by a 2D max pooling layer in the encoding sub-network. In the decoding sub-network, 3 upsampling blocks are composed of a upsample2D layer followed by a 2D convolution, a concatenation operation with the residual connection and two 2D convolutions.

## Default Configuration
This model trains in 2500 epochs, under the following setup:

> - Global Batch Size: 16
> - Optimizer RMSProp:
     decay: 0.9
     momentum: 0.8
     centered: True
> - Learning Rate Schedule: Exponential Step Decay
    decay: 0.8
    steps: 500
    initial learning rate: 1e-4
> - Weight Initialization: He Uniform Distribution (introduced by Kaiming He et al. in 2015 to address issues related ReLU activations in deep neural networks)
> - Loss Function:
    When DICE Loss < 0.3, Loss = Binary Cross Entropy
    Else, Loss = DICE Loss
> - Data Augmentation
    Random Horizontal Flip (50% chance)
    Random Rotation 90Â°
> - Activation Functions:
   ReLU is used for all layers
   Sigmoid is used at the output to ensure that the ouputs are between [0, 1]
> - Weight decay: 1e-5

## Features
### Automatic Mixed Precision (AMP)

This implementation of UNet uses AMP to implement mixed precision training. It allows us to use FP16 training with FP32 master weights by modifying just a few lines of code.

### Horovod

Horovod is a distributed training framework for TensorFlow, Keras, PyTorch, and MXNet. The goal of Horovod is to make distributed deep learning fast and easy to use. For more information about how to get started with Horovod, see the Horovod: Official repository.

### Multi-GPU Training with Horovod

Our model uses Horovod to implement efficient multi-GPU training with NCCL


## Requirements
This repository contains Dockerfile which extends the TensorFlow NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

> - NVIDIA Docker
> - TensorFlow NGC container
> - GPU-based architecture:
> > - NVIDIA Volta
> > - NVIDIA Turing
> > - NVIDIA Ampere architecture

Follow the rest of the process from building the container to train and test the model in quick start guide section.
