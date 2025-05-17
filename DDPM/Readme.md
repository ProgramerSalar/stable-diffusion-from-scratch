# Denoising Diffusion Probabilistic Models (DDPM)

This repository contains a complete implementation of the Denoising Diffusion Probabilistic Models (DDPM) as described in the original paper. The project is designed to train a diffusion model for image generation using a dataset of cat and dog images.

![DDPM Image](Image/denoising-diffusion.png)

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [File Structure](#file-structure)
4. [Requirements](#requirements)
5. [Setup](#setup)
6. [Usage](#usage)
    - [Training](#training)
    - [Sampling](#sampling)
7. [Configuration](#configuration)
8. [Dataset](#dataset)
9. [Model Checkpoints](#model-checkpoints)
10. [Code Overview](#code-overview)
    - [Key Components](#key-components)
    - [Pipeline](#pipeline)
    - [Model Architecture](#model-architecture)
11. [References](#references)
12. [License](#license)

---

## Introduction

Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that learn to generate data by reversing a diffusion process. The diffusion process gradually adds noise to the data, and the model is trained to reverse this process step by step. This repository implements the DDPM framework using PyTorch and provides tools for training, evaluation, and sampling.

---

## Features

- **Forward and Reverse Diffusion Processes**: Implements the forward diffusion process to add noise and the reverse process to denoise.
- **UNet Architecture**: Uses a UNet-based model for noise prediction.
- **Configurable Training Pipeline**: Includes support for checkpoints, evaluation, and sampling.
- **Dataset Loader**: Preprocessing and augmentation for image datasets.
- **Sampling Animations**: Visualizes the diffusion process as a GIF.

---

## File Structure

```text
DDPM/
│
├── config/
│   ├── config.py                 # Configuration file for training
│   
├── models/
│   ├── layers.py                 # Building blocks for the UNet model
│   ├── unet.py                   # UNet model implementation
│       
├── dataset/
│   ├── cat_dog_images/           # Dataset folder containing cat and dog images
│
├── checkpoint/                   # Folder for saving model checkpoints
│
├── Ganerated_Image/
│   ├── samples/                  # Folder for generated images and animations
│
├── Image/                        # Folder for storing static images
│   ├── denoising-diffusion.png   # Diagram of the DDPM process
│
├── ImageDataset.py               # Dataset loader and preprocessing
├── train.py                      # Training script
├── utils.py                      # Utility functions for image processing
├── ddpm.py                       # DDPM pipeline implementation
├── .gitignore                    # Git ignore file
├── README.md                     # Project documentation
```

---

## Requirements

- Python 3.9 or higher
- PyTorch
- torchvision
- matplotlib
- tqdm
- PIL (Pillow)

Install the required packages using:
```bash
pip install -r requirements.txt
```

---

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/ddpm.git
    cd ddpm
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Prepare the dataset by placing images in the `dataset/cat_dog_images/` folder.

---

## Usage

### Training

To train the model, run the following command:
```bash
python train.py
```

The training script will:
- Load the dataset from the `dataset/cat_dog_images/` folder.
- Train the UNet model using the DDPM framework.
- Save model checkpoints in the `checkpoint/` directory.
- Generate and save sampled images in the `Ganerated_Image/samples/` directory.

### Sampling

To generate images using the trained model, run:
```bash
python ddpm.py
```

The sampling process generates images from random noise using the trained model. The generated images and animations are saved in the `Ganerated_Image/samples/` directory.

---

## Configuration

The training and evaluation parameters are defined in [config/config.py](config/config.py). Key parameters include:

- `image_size`: Size of the input images (default: 256x256).
- `image_channels`: Number of image channels (default: 3 for RGB).
- `train_batch_size`: Batch size for training.
- `eval_batch_size`: Batch size for evaluation.
- `num_epochs`: Number of training epochs.
- `learning_rate`: Learning rate for the optimizer.
- `diffusion_timesteps`: Number of timesteps in the diffusion process.
- `output_dir`: Directory for saving generated images and animations.
- `device`: Device for training (e.g., "cuda" or "cpu").

---

## Dataset

The dataset should be placed in the `dataset/cat_dog_images/` folder. It should contain images in `.jpg`, `.jpeg`, `.png`, or `.bmp` formats. The dataset loader in `ImageDataset.py` handles preprocessing and augmentation.

---

## Model Checkpoints

Model checkpoints are saved in the `checkpoint/` directory during training. Each checkpoint contains:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Training configuration
- Current epoch

To resume training from a checkpoint, set the `resume` parameter in the configuration file to the path of the checkpoint.

---

## Code Overview

### Key Components

1. **[ddpm.py](ddpm.py)**: Implements the DDPM pipeline, including forward and reverse diffusion processes, and sampling.
2. **[models/unet.py](models/unet.py)**: Defines the UNet model architecture.
3. **[models/layers.py](models/layers.py)**: Contains building blocks for the UNet model, including convolutional and attention layers.
4. **[ImageDataset.py](ImageDataset.py)**: Handles dataset loading, preprocessing, and augmentation.
5. **[train.py](train.py)**: Training script for the DDPM model.
6. **[utils.py](utils.py)**: Utility functions for image processing and visualization.

---

### DDPMPipeline Class Functionality

The `DDPMPipeline` class, implemented in **[ddpm.py](ddpm.py)**, is the core of the diffusion process. It handles forward diffusion, reverse diffusion, and sampling. Below is an overview of its functionality:

#### Key Methods in the `DDPMPipeline` Class:
1. **`__init__`**:
   - Initializes the DDPM pipeline with parameters such as:
     - `beta_start`: Starting value for the beta schedule.
     - `beta_end`: Ending value for the beta schedule.
     - `num_timesteps`: Number of timesteps in the diffusion process.
   - Precomputes `alphas` and `alphas_hat` for efficient computation.

2. **`forward_diffusion(images, timesteps)`**:
   - Implements the forward diffusion process by adding noise to the input images.
   - Uses Equation (14) from the DDPM paper.
   - Returns the noisy images and the added Gaussian noise.

3. **`reverse_diffusion(model, noisy_images, timesteps)`**:
   - Implements the reverse diffusion process to predict and remove noise step by step.
   - Uses the provided model (e.g., UNet) to predict the noise.

4. **`sampling(model, initial_noise, device, save_all_steps=False)`**:
   - Implements the sampling process (Algorithm 2 from the DDPM paper).
   - Starts from random noise and iteratively denoises it to generate images.
   - Parameters:
     - `model`: The trained noise prediction model (e.g., UNet).
     - `initial_noise`: The starting noise tensor.
     - `device`: The device to perform computations on (e.g., "cuda" or "cpu").
     - `save_all_steps`: If `True`, saves all intermediate steps of the sampling process.
   - Returns the final generated image or all intermediate steps.

---

### Example Usage of the `DDPMPipeline` Class

Here is an example of how to use the `DDPMPipeline` class in your code:

```python
# filepath: [ddpm.py](http://_vscodecontentref_/2)
import torch
from ddpm import DDPMPipeline
from models.unet import UNet

# Initialize the UNet model
model = UNet(image_channels=3, base_channels=64)

# Initialize the DDPM pipeline
ddpm = DDPMPipeline(beta_start=1e-4, beta_end=1e-2, num_timesteps=1000)

# Forward diffusion (training)
images = torch.randn(16, 3, 256, 256)  # Example batch of images
timesteps = torch.randint(0, 1000, (16,))  # Random timesteps for the batch
noisy_images, noise = ddpm.forward_diffusion(images, timesteps)

# Reverse diffusion (sampling)
device = "cuda" if torch.cuda.is_available() else "cpu"
initial_noise = torch.randn(16, 3, 256, 256).to(device)  # Starting noise
generated_images = ddpm.sampling(model, initial_noise, device)

# Compute loss during training
predicted_noise = model(noisy_images, timesteps)
loss = torch.nn.functional.mse_loss(predicted_noise, noise)
```

### Pipeline

The DDPM pipeline consists of:
1. **Forward Diffusion**: Adds noise to the input images over a series of timesteps.
2. **Reverse Diffusion**: Predicts and removes noise step by step to generate images.

### Model Architecture

The UNet model is used for noise prediction. It consists of:
- Downsampling blocks with convolutional and attention layers.
- A bottleneck layer with self-attention.
- Upsampling blocks with convolutional and attention layers.


### Explanation:
- The **Key Methods** section explains the purpose of each method in the [DDPMPipeline](https://github.com/ProgramerSalar/DDPM/blob/master/ddpm.py) class.
- The **Example Usage** section demonstrates how to use the class for forward diffusion, reverse diffusion, and sampling.
- The **Pipeline** section summarizes the overall process.

You can add this updated section to your Readme file under the **Code Overview** section.

---

## References

- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [PyTorch Documentation](https://pytorch.org/docs/)

### DDPM YouTube Video

[![DDPM YouTube Video](https://img.youtube.com/vi/2zCcucNqWIQ/0.jpg)](https://www.youtube.com/watch?v=2zCcucNqWIQ&t=9153s)

Click the thumbnail above to watch the video.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


