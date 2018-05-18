# Image denoising and completion neural network with tensor flow [![GitHub license](https://img.shields.io/github/license/nea/MarkdownViewerPlusPlus.svg)](https://github.com/nea/MarkdownViewerPlusPlus/blob/master/LICENSE.md) [![GitHub (pre-)release](https://img.shields.io/badge/release-0.8.2-yellow.svg)](https://github.com/nea/MarkdownViewerPlusPlus/releases/tag/0.8.2) [![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.me/insanitydesign)

This is an example of neural network made for image denoising and completion. 
The network architecture novel but inspired by existing convolution neural networks (CNN) made for image segmentation.
Other sources of inspiration for the architecture is res-Net and Multigrid from numerical analysis.

Beside being a specific neural net architecture the code is written is a way that allows to configure the network architecture from a configuration file.
The project also contains a general framework for training and visualizing any type of CNN. 

## Features
* Configuration file to allow control on the training and some of the architecture. 
Also allows control on the data paths and training parameters (such as rate, optimizer, etc.).
* Module to get prediction on already trained network to visualize results.

## Prerequisites
For my development I used windows with python 3.6 and numpy and TensorFlow. 
However the code should be compatible to any operating system and python version as long as TensorFlow version is above 1.6.
* Python 3.6.
* TensorFlow 1.8.0.
* numpy 1.4.
* Project folder should be added to Python Path.