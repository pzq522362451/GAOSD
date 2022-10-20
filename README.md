# GAOSD

## One-shot learning with Generative adversarial network

Code accompanying the paper "Generative Adversarial One-Shot Diagnosis of Transmission Faults for Industrial Robots" by Authors (Ready to be submitted for publication).

-  Tensorflow 1.15.0 implementation
-  Inspired by Jeff Donahue $et$ $al$. [Adversarial feature learning] (https://arxiv.org/pdf/1605.09782.pdf)(Bi-GAN)
-  This repository contains several experiments mentioned in the paper
-  The proposed GAOSD was verified with the local six-degree-of-freedom industrial robot dataset.
-  One of the implementations for Bi-GAN using the MNIST dataset is shown at (https://github.com/jeffdonahue/bigan)

## Requirements

- python 3.7.13
- Tensorflow == 1.15.0
- Numpy == 1.19.5
- Keras == 2.3.1

Note: All experiment were excecuted in Google colab with Tesla T4 GPU ![alt text](https://colab.research.google.com/assets/colab-badge.svg)


## Main file discription
* `--main`: The GAOSD model we build for runing some experiments. It is a class and based on tensorflow 1.15.0.
* `--main_saprseae`: Main Functions about Sparse auto-encoder.
* `--main_dcae`: Main Functions about deep convolutional auto-encoder.
* `--encoder_bigan`: To project the dataset into the features space with a trained encoder from Bi-GAN (main.py).
* `--encoder_dcae`: To project the dataset into the features space with a trained encoder from the deep convolutional auto-encoder (main_dcae.py).
* `--encoder_sae`: To project the dataset into the features space with a trained encoder from the sparse auto-encoder (main_saprseae.py).
* `--encoder_wpt`: To project the dataset into the featurets space of Wavelet packet transform (wpt).
* `--model`:  Model architectures

## Implementation details
- The overall experiments include GAOSD,OSD-SAE,OSD-DCAE and OSD-WFE are included in src. Note that users should change the directory to successfully run this code.
- Hyperparameter settings: Adam optimizer is used with learning rate of `2e-4` in both the generator and the discriminator;The batch size is `64`, total iteration for Bi-GAN is 1000. For the random forest, 100 nodes were chosen with their default setting for running 100 times to get an optimal result.  

