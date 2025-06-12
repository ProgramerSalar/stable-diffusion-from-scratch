# Encoder-Decoder for Latent Diffusion Models

This module implements a **modular Encoder and Decoder architecture** in PyTorch, designed for use in latent diffusion models and other generative frameworks. The code is inspired by the architectures used in modern generative models such as Stable Diffusion, VQ-VAE, and autoencoders for image synthesis.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
  - [Encoder Example](#encoder-example)
  - [Decoder Example](#decoder-example)
- [Architecture Details](#architecture-details)
  - [Encoder](#encoder)
  - [Decoder](#decoder)
- [Customization](#customization)
- [Extending the Model](#extending-the-model)


---

## Overview

This file provides two main classes:

- [`Encoder`](encoder.py): Compresses an input image into a lower-dimensional latent representation.
- [`Decoder`](encoder.py): Reconstructs an image from a latent representation.

Both classes are highly configurable and support residual blocks, attention mechanisms (vanilla and linear), and flexible channel scaling. These components are essential for building powerful and scalable autoencoders and diffusion models.

---

## Features

- **PyTorch Implementation**: Fully implemented using PyTorch, easy to integrate with other PyTorch-based projects.
- **Residual Blocks**: Uses ResNet-style blocks for stable training and deep architectures.
- **Attention Support**: Includes both standard ("vanilla") and efficient ("linear") attention mechanisms.
- **Flexible Architecture**: Easily configurable number of channels, resolution, attention resolutions, and more.
- **Latent Space**: Supports configurable latent dimensionality for use in VAE, VQ-VAE, or diffusion models.
- **Modular Design**: Components such as attention, normalization, and up/downsampling are modular and reusable.

---

## File Structure

```
encoder.py # Encoder and Decoder classes
```

## Usage

### Encoder Example

```python

import torch
from encoder import Encoder

x = torch.randn(1, 3, 128, 128)  # Example input image

encoder = Encoder(
    ch=64,
    out_ch=3,
    ch_mult=(1, 1, 2, 2, 4),
    num_res_blocks=2,
    attn_resolutions=[16],
    in_channels=3,
    resolution=128,
    z_channels=16
)

output = encoder(x)
print(output.shape)  # Output: torch.Size([1, 16, 8, 8])
```

### Decoder Example 

```python

import torch
from encoder import Decoder

z = torch.randn(1, 16, 8, 8)  # Example latent code

decoder = Decoder(
    ch=64,
    out_ch=3,
    ch_mult=(1, 1, 2, 2, 4),
    num_res_blocks=2,
    attn_resolutions=[16],
    in_channels=3,
    resolution=128,
    z_channels=16
)

recon = decoder(z)
print(recon.shape)  # Output: torch.Size([1, 3, 128, 128])
```

## Architecture Details

### Encoder

- **Input**: Image tensor of shape `[batch, in_channels, resolution, resolution]`.

- **Downsampling**: Series of residual blocks and optional attention layers, with spatial downsampling at each stage.

- **Latent Output**: Final output is a latent tensor of shape [batch, z_channels, latent_h, latent_w], where latent_h and latent_w depend on the number of downsampling stages.

**Key Parameters**

- `ch`: Base channel count.
- `out_ch`: Number of output channels (for compatibility, usually matches z_channels).
- `ch_mult`: Tuple of multipliers for each resolution level.
- `num_res_blocks`: Number of residual blocks per resolution.
- `attn_resolutions`: List of resolutions at which to apply attention.
- `in_channels`: Number of input channels (e.g., 3 for RGB).
- `resolution`: Input image resolution (must be divisible by 2^len(ch_mult)).
- `z_channels`: Number of latent channels.
- `attn_type`: Type of attention ("vanilla" or "linear").


## Decoder

- **Input**: Latent tensor of shape [batch, z_channels, latent_h, latent_w].
- **Upsampling**: Series of residual blocks and optional attention layers, with spatial upsampling at each stage.
- **Image Output**: Final output is an image tensor of shape [batch, out_ch, resolution, resolution].


**Key Parameters**

- `ch`: Base channel count.
- `out_ch`: Number of output channels (e.g., 3 for RGB).
- `ch_mult`: Tuple of multipliers for each resolution level.
- `num_res_blocks`: Number of residual blocks per resolution.
- `attn_resolutions`: List of resolutions at which to apply attention.
- `in_channels`: Number of input channels (for compatibility, usually matches z_channels).
- `resolution`: Output image resolution.
- `z_channels`: Number of latent channels.
- `attn_type`: Type of attention ("vanilla" or "linear").
- `tanh_out`: If True, applies tanh to the output (useful for image normalization).

## Customization
You can customize the encoder and decoder by changing the following parameters:

- `ch`: Base channel count.
- `out_ch`: Number of output channels.
- `ch_mult`: Tuple of multipliers for each resolution level.
- `num_res_blocks`: Number of residual blocks per resolution.
- `attn_resolutions`: List of resolutions at which to apply attention.
- `dropout`: Dropout rate in residual blocks.
- `resamp_with_conv`: Whether to use convolution in up/downsampling.
- `in_channels`: Number of input channels.
- `resolution`: Input/output image resolution.
- `z_channels`: Number of latent channels.
- `attn_type`: Type of attention ("vanilla", "linear", or "none").
- `tanh_out`: (Decoder only) Whether to apply tanh to the output.


## Extending the Model

- **Add More Attention Types**: Implement new attention mechanisms in `Unet/attention.py` and update `make_attention`.
- **Conditional Inputs**: Pass additional context to the model via the forward method.
- **Different Normalization**: Swap out `Normalize` for other normalization layers as needed.
- **Custom Residual Blocks**: Extend `ResnetBlock` for more complex architectures.

