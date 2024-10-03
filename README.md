# Image Classification using CIFAR-10

This project demonstrates an image classification pipeline using the CIFAR-10 dataset, implemented with PyTorch.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)

## Project Overview

This repository implements an image classification system using the CIFAR-10 dataset. The model is built using PyTorch, and the dataset is loaded using the Hugging Face Datasets library.

## Features

- Image classification using the CIFAR-10 dataset.
- PyTorch-based model training.
- Customizable image transformations and preprocessing.
- Easy extensibility for other datasets or classification tasks.

## Technologies Used

- **Python**
- **PyTorch**: For building and training the neural network.
- **Hugging Face Datasets**: For easy access to CIFAR-10.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/ycheung5/image-classification
    cd image-classification
    ```

2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the first cell to import dependencies:**
    ```bash
    from PIL import Image

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    
    from torch.utils.data import DataLoader
    
    import torchvision.transforms as transforms
    ```
    
4. **Put your images path in the image_paths array in the last cell**

- **Get Final Result:**
    - Once the task is complete, you can see the final result in the output cell.

## Dataset

The CIFAR-10 dataset is a collection of images categorized into 10 different classes. The dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. 

You can explore more about the dataset on the [CIFAR-10 dataset page]([https://huggingface.co/datasets/uoft-cs/cifar10]).

## Training

The PyTorch model can be trained with the following steps:

- Load the CIFAR-10 dataset from Hugging Face:
  ```python
  from datasets import load_dataset
  ds = load_dataset("cifar10")
