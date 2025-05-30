# Variational Autoencoder (VAE) PyTorch Implementation

This module provides a PyTorch Lightning implementation of a **Variational Autoencoder (VAE)** for image data, using a modular encoder-decoder architecture inspired by modern generative models. The code is designed for flexibility, research, and educational purposes, and supports easy customization of model depth, attention, and latent space dimensionality.

---

## Table of Contents

1. Overview
2. Features
3. File Structure
4. Installation
5. Usage
    - Training
    - Validation
    - Sampling
6. Architecture
    - Encoder
    - Decoder
    - Latent Space
    - Loss Function
7. Customization
8. API Reference
9. Example
10. Extending the Model
11. License

---

## Overview

The `AutoEncoderKL` class implements a VAE for images, using configurable encoder and decoder modules. The model supports:

- Flexible channel scaling and resolution
- Residual blocks and attention mechanisms
- KL-divergence regularization in the latent space
- Modular loss configuration
- PyTorch Lightning integration for easy training and validation

---

## Features

- **PyTorch Lightning Integration**: Simplifies training, validation, and logging.
- **Residual Blocks**: For stable and deep architectures.
- **Attention Support**: Includes both standard ("vanilla") and efficient ("linear") attention mechanisms.
- **Flexible Latent Space**: Configurable latent dimensionality for use in VAE, VQ-VAE, or diffusion models.
- **Modular Design**: Encoder and decoder are modular and reusable.
- **Custom Loss Support**: Plug in your own loss function via configuration.
- **Mixed Precision & GPU Support**: Ready for fast training on modern hardware.

---

## File Structure

```
VAE/
├── autoencoder.py        # Main VAE implementation (this file)
├── loss.py
Encoder_Decoder/
├── encoder.py            # Encoder and Decoder modules
Distribution/
├── distribution.py       # DiagonalGaussianDistribution for VAE latent space
config/
├── config.yaml           # Example configuration file
```

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/stable-diffusion-from-scratch.git
    cd stable-diffusion-from-scratch
    ```

2. Install dependencies:
    ```bash
    pip install req.txt
    ```

---

## Usage

### Training

To train the VAE, run:

```bash
python VAE/autoencoder.py
```

This will:
- Load configuration from [`config/config.yaml`](stable-diffusion-from-scratch/AutoEncoder/autoencoder.py )
- Initialize the model and dataloaders
- Train for the specified number of epochs
- Log losses and metrics

### Validation

Validation is performed automatically at the end of each epoch. The validation loss is logged and can be monitored for overfitting.

### Sampling

After training, you can sample from the latent space by passing random tensors through the decoder:

```python
z = torch.randn(batch_size, z_channels, latent_h, latent_w).to(device)
samples = autoencoder.decode(z)
```

---

## Architecture

### Encoder

The encoder compresses an input image into a lower-dimensional latent representation. It consists of:

- Initial convolution
- Multiple downsampling stages with residual blocks and optional attention
- Middle bottleneck with residual and attention blocks
- Output convolution to produce latent moments (mean and log-variance)

### Decoder

The decoder reconstructs an image from a latent code. It consists of:

- Initial convolution from latent space
- Multiple upsampling stages with residual blocks and optional attention
- Middle bottleneck with residual and attention blocks
- Output convolution to produce the reconstructed image

### Latent Space

The latent space is modeled as a diagonal Gaussian distribution. The encoder outputs mean and log-variance, and the latent code is sampled using the reparameterization trick.

### Loss Function

The default loss is the sum of:

- **Reconstruction Loss**: Mean squared error (MSE) between input and reconstruction.
- **KL Divergence**: Regularizes the latent space to match a standard normal distribution.

Total loss:
```
total_loss = rec_loss + 0.5 * kl_loss
```

---

## Customization

You can customize the model via the [`ddconfig`](stable-diffusion-from-scratch/AutoEncoder/autoencoder.py ) dictionary or YAML config file:

- [`ch`](stable-diffusion-from-scratch/Encoder_Decoder/encoder.py ): Base channel count.
- [`out_ch`](stable-diffusion-from-scratch/Encoder_Decoder/encoder.py ): Number of output channels (e.g., 3 for RGB).
- [`ch_mult`](stable-diffusion-from-scratch/Encoder_Decoder/encoder.py ): Tuple of multipliers for each resolution level.
- [`num_res_blocks`](stable-diffusion-from-scratch/Encoder_Decoder/encoder.py ): Number of residual blocks per resolution.
- [`attn_resolutions`](stable-diffusion-from-scratch/Encoder_Decoder/encoder.py ): List of resolutions at which to apply attention.
- [`dropout`](stable-diffusion-from-scratch/Encoder_Decoder/encoder.py ): Dropout rate in residual blocks.
- [`resamp_with_conv`](stable-diffusion-from-scratch/Encoder_Decoder/encoder.py ): Whether to use convolution in up/downsampling.
- [`in_channels`](/home/manish/anaconda3/envs/cuda121/lib/python3.10/site-packages/torch/nn/modules/conv.py ): Number of input channels.
- [`resolution`](stable-diffusion-from-scratch/Encoder_Decoder/encoder.py ): Input/output image resolution.
- [`z_channels`](stable-diffusion-from-scratch/Encoder_Decoder/encoder.py ): Number of latent channels.
- [`double_z`](stable-diffusion-from-scratch/Encoder_Decoder/encoder.py ): Whether to double latent channels for mean and logvar.
- [`attn_type`](stable-diffusion-from-scratch/Encoder_Decoder/encoder.py ): Type of attention ("vanilla", "linear", or "none").

Example config:
```yaml
model:
  params:
    ddconfig:
      ch: 128
      out_ch: 3
      ch_mult: [1, 1, 2, 2, 4]
      num_res_blocks: 2
      attn_resolutions: [16]
      in_channels: 3
      resolution: 256
      double_z: True
      z_channels: 16
    embed_dim: 16
```

---

## API Reference

### [`AutoEncoderKL`](stable-diffusion-from-scratch/AutoEncoder/autoencoder.py ) class

**Constructor**
```python
AutoEncoderKL(ddconfig, embed_dim, lossconfig=None, ckpt_path=None, ignore_keys=[], colorize_nlabels=None, monitor=None)
```

**Key Methods**
- [`forward(input, sample_posterior=True)`](stable-diffusion-from-scratch/AutoEncoder/autoencoder.py ): Returns reconstruction and posterior for input batch.
- [`encode(x)`](stable-diffusion-from-scratch/AutoEncoder/autoencoder.py ): Returns posterior distribution for input.
- [`decode(z)`](stable-diffusion-from-scratch/AutoEncoder/autoencoder.py ): Decodes latent code to image.
- [`training_step(batch, batch_idx)`](stable-diffusion-from-scratch/VAE/autoencoder.py ): Training step for PyTorch Lightning.
- [`validation_step(batch, batch_idx)`](stable-diffusion-from-scratch/VAE/autoencoder.py ): Validation step for PyTorch Lightning.
- [`configure_optimizers()`](stable-diffusion-from-scratch/VAE/autoencoder.py ): Returns optimizer(s) for training.
- [`default_loss(inputs, reconstructions, posterior, global_step, last_layer, split)`](stable-diffusion-from-scratch/VAE/autoencoder.py ): Default VAE loss.

---

## Example

```python
import torch
from VAE.autoencoder import AutoEncoderKL
from Encoder_Decoder.encoder import Encoder, Decoder

# Example config
ddconfig = {
    "ch": 128,
    "out_ch": 3,
    "ch_mult": (1, 1, 2, 2, 4),
    "num_res_blocks": 2,
    "attn_resolutions": [16],
    "in_channels": 3,
    "resolution": 256,
    "double_z": True,
    "z_channels": 16
}

autoencoder = AutoEncoderKL(ddconfig=ddconfig, embed_dim=16)
x = torch.randn(4, 3, 256, 256)
recon, posterior = autoencoder(x)
print("Reconstruction shape:", recon.shape)
```

---

## Extending the Model

- **Custom Loss**: Pass a custom loss function via [`lossconfig`](stable-diffusion-from-scratch/AutoEncoder/autoencoder.py ).
- **Discriminator**: Add adversarial loss by uncommenting and extending the relevant sections in the code.
- **Attention Types**: Implement new attention mechanisms in `Unet/attention.py` and update the encoder/decoder.
- **Conditional Inputs**: Pass additional context to the model via the forward method.
- **Different Normalization**: Swap out [`Normalize`](stable-diffusion-from-scratch/Unet/unet.py ) for other normalization layers as needed.
- **Custom Residual Blocks**: Extend [`ResnetBlock`](stable-diffusion-from-scratch/Unet/unet.py ) for more complex architectures.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

This implementation is inspired by modern generative models such as Stable Diffusion, VQ-VAE, and research on variational autoencoders.

---

**For more details, see the source code in [`stable-diffusion-from-scratch/AutoEncoder/autoencoder.py`](https://github.com/ProgramerSalar/stable-diffusion-from-scratch/blob/master/VAE/autoencoder.py).**