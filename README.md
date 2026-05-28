# Diffusion-Based Methods for Inverse Problems in Physical Data

**Skoltech Deep Learning Final Project**

## Overview

This project applies diffusion models as data-driven priors to solve inverse problems in physical fields. Given sparse, masked, downsampled, or blurred observations of a 2D turbulent velocity field, we reconstruct the full field in a physically consistent way — without retraining the model for each observation operator.

The core inverse problem:

```
y = A(u) + η,    η ~ N(0, σ²I)
```

recover the full velocity field **u** from incomplete, possibly noisy measurements **y**.

## Dataset

We use **Kolmogorov flow** — 2D incompressible Navier-Stokes with sinusoidal forcing, simulated on a 256×256 periodic grid and average-pooled to 64×64. Velocity fields (u_x, u_y) are stored as tensors of shape `[N, 2, 64, 64]`.

Four observation operators are evaluated:

| Operator | Description | Coverage |
|---|---|---|
| Sparse grid | Regular 4×4 stride mask | ~6% known |
| Box mask | Central 32×32 region known | 25% known |
| Downsample | 4× average pooling (64→16) | 6.25% |
| Gaussian blur | Periodic circular blur + observation noise | full size, degraded |

## Model

A **VP-SDE score network** (`diffusers.UNet2DModel`) is trained unconditionally on clean velocity snapshots:

- **Input:** 6 channels — 2 noisy velocity channels + 4 Fourier coordinate channels (sin x, cos x, sin y, cos y)
- **Architecture:** 3-level UNet, channel widths 96→192→384, attention on levels 2–3, circular padding for periodic boundaries
- **Training:** AdamW, lr=2×10⁻⁴, BF16, batch size 128, 1024 epochs on NVIDIA A100; EMA weights used for inference

## Conditional Sampling Methods

Four zero-shot methods inject measurement information at inference time without modifying the score model:

| Method | Key idea | Applicable operators |
|---|---|---|
| **DDNM** | Null-space decomposition; replaces the range component of the predicted clean field with the pseudoinverse of the observation | All four |
| **DPS** | Gradient guidance through the predicted clean field; works for any differentiable A | All four |
| **RePaint** | Pastes the observation (noised to the current diffusion level) into the known region at every reverse step | Mask operators only |
| **DDRM** | Per-frequency Bayesian update in the Fourier/SVD basis | Downsample, blur |

## Physics Constraint

Incompressibility (∇·u = 0) is enforced via **Helmholtz projection** in Fourier space after each data-consistency correction. This is training-free and consistently reduces divergence error across all methods.

## Key Results

| Operator | Method | Rel-L₂ | Divergence |
|---|---|---|---|
| Sparse grid | DPS | 0.198 | 0.038 |
| Downsample | DDNM | 0.050 | 0.011 |
| Blur | DDRM | 0.079 | 0.182 |

Helmholtz projection reduces divergence for DDNM, RePaint, and DDRM with minimal accuracy penalty.

## Contributors

| Name | Contribution |
|---|---|
| Mariia Abdullaeva | RePaint sampler |
| Andrei Efimov | DDNM sampler |
| Anton Losev | DDRM sampler |
| Amir Sadreev | Model training and DPS sampler |

## References

1. Chung et al. *Diffusion Posterior Sampling for General Noisy Inverse Problems.* ICLR 2023.
2. Kawar et al. *Denoising Diffusion Restoration Models.* NeurIPS 2022.
3. Lugmayr et al. *RePaint: Inpainting using Denoising Diffusion Probabilistic Models.* CVPR 2022.
4. Wang et al. *Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model.* ICLR 2023.
