# LitEma: Exponential Moving Average for PyTorch Models

This repository contains an implementation of the `LitEma` class, which provides a simple and efficient way to maintain Exponential Moving Averages (EMA) of model parameters during training. EMA is a common technique used in deep learning to stabilize training and improve generalization by maintaining a smoothed version of the model's weights.

---

## Features

- **EMA Updates**: Automatically updates shadow parameters using EMA after each training step.
- **Parameter Restoration**: Temporarily replaces model parameters with EMA weights for evaluation or checkpoint saving.
- **Customizable Decay**: Supports dynamic decay adjustment based on the number of updates.
- **Integration with PyTorch**: Fully compatible with PyTorch models and training workflows.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [How It Works](#how-it-works)
4. [API Reference](#api-reference)
5. [Example](#example)
6. [License](#license)

---

## Installation

To use the `LitEma` class, clone this repository and ensure you have PyTorch installed:

```bash
git clone https://github.com/your-repo/stable-diffusion.git
cd stable-diffusion
pip install torch
```

## Usage
Initializing LitEma

To use LitEma, initialize it with your model and desired decay rate:

```python
from ema import LitEma

model = YourModel()
ema = LitEma(model, decay=0.999)
```

Updating EMA Weights

Call the ema object after each optimization step to update the EMA weights:

```python
optimizer.step()
ema(model)
```

Using EMA Weights for Evaluation

To evaluate the model using EMA weights:

```python
ema.store(model.parameters())  # Store original parameters
ema.copy_to(model)            # Copy EMA weights to the model

# Perform evaluation
model.eval()
with torch.no_grad():
    outputs = model(inputs)

ema.restore(model.parameters())  # Restore original parameters
```


### How It Works
The LitEma class maintains a shadow copy of the model's parameters and updates them using the following formula:

```bash
shadow = shadow - (1 - decay) * (shadow - current)
```

Where:

- `shadow` is the EMA parameter.
- `current` is the current model parameter.
- `decay` is the smoothing factor (close to 1).
The class also supports dynamic decay adjustment based on the number of updates, which can help stabilize training in the early stages.


## API Reference

`LitEma`

**Constructor**

```bash
LitEma(model, decay=0.999, use_num_updates=True)
```

- **model**: The PyTorch model whose parameters will be tracked.
- **decay**: The decay rate for EMA updates (default: `0.999`).
- **use_num_updates**: Whether to adjust decay dynamically based on the number of updates (default: True).


**Methods**

- `reset_num_updates()`: Resets the update counter to zero.
- `forward(model)`: Updates the EMA weights using the current model parameters.
- `copy_to(model)`: Copies the EMA weights to the given model.
- `store(parameters)`: Stores the current model parameters.
- `restore(parameters)`: Restores the previously stored parameters.


## Example

Below is a complete example of using `LitEma` with a simple PyTorch model:


```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ema import LitEma

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Create synthetic data
x = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, optimizer, and EMA
model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
ema = LitEma(model, decay=0.999)

# Training loop
for epoch in range(10):
    model.train()
    for batch_x, batch_y in dataloader:
        outputs = model(batch_x)
        loss = nn.MSELoss()(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA weights
        ema(model)

    # Validation with EMA weights
    ema.store(model.parameters())
    ema.copy_to(model)
    model.eval()
    with torch.no_grad():
        val_outputs = model(x)
        val_loss = nn.MSELoss()(val_outputs, y)
        print(f"Epoch {epoch + 1}, EMA Validation Loss: {val_loss.item():.4f}")
    ema.restore(model.parameters())
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## cknowledgments
This implementation is inspired by common practices in deep learning frameworks and research papers that utilize Exponential Moving Averages for model stabilization and performance improvement. ```