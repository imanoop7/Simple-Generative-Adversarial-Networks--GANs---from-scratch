# Simple Generative Adversarial Networks (GANs) from Scratch

This repository contains implementations of Generative Adversarial Networks (GANs) from scratch using both NumPy and PyTorch. The implementations are based on the original GAN paper by Goodfellow et al. (2014).

## Implementations

1. `GAN.py` - Pure NumPy implementation
   - Implements GAN using only NumPy for better understanding of the underlying mathematics
   - Includes manual gradient calculations
   - Suitable for learning GAN fundamentals

2. `GAN-pytorch.py` - PyTorch implementation
   - Modern implementation using PyTorch
   - Utilizes automatic differentiation
   - GPU acceleration support
   - More efficient and scalable

## Requirements

```bash
numpy>=1.19.2
torch>=1.7.0
matplotlib>=3.3.2
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the NumPy version:
```bash
python GAN.py
```

Run the PyTorch version:
```bash
python GAN-pytorch.py
```

Both implementations will:
- Train a GAN to generate samples from a Gaussian distribution
- Display training progress
- Show histograms of generated samples

## Architecture

Both implementations include:
- Generator: Transforms random noise into fake data
- Discriminator: Distinguishes between real and fake data
- Training loop with gradient updates
- Visualization of results

## Reference

[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) by Goodfellow et al.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
