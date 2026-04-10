# DCGAN — Realistic Face Image Generation

Deep Learning Assignment-3: Train a GAN model to generate realistic face images using the CelebA-HQ dataset.

## Quick Start

```bash
# Activate the conda environment (required for heavy computation)
conda activate dl_assignment

# 1. Train the DCGAN model (50 epochs, ~35 min on GPU)
python train_gan.py

# 2. Generate 15 synthetic face images
python generate_images.py

# 3. Evaluate image quality (FID score)
python evaluate.py
```

## Project Structure

```
project/
├── data/                       # CelebA-HQ dataset (30,000 images @ 1024×1024)
│   └── image/                  # Face images: 0.jpg ... 29999.jpg
├── models/                     # Saved checkpoints
│   ├── checkpoint_epoch_XXX.pth
│   ├── generator_best.pth
│   └── discriminator_best.pth
├── outputs/                    # Training visualizations
│   ├── generated_epoch_XXX.png # Sample grids at each checkpoint
│   └── generated_grid.png      # Final composite grid
├── samples/                    # Generated face images
│   ├── sample_1.png
│   ├── sample_2.png
│   └── ... (15 total)
├── logs/
│   ├── train.log               # Training log with per-epoch metrics
│   └── loss_history.json       # Full loss curves data
├── metrics/
│   └── metrics.json            # FID score and evaluation metrics
├── report/
│   └── assignment_report.txt   # Full assignment report
├── config.py                   # All hyperparameters and paths
├── utils.py                    # Helper functions
├── train_gan.py                # Training script
├── generate_images.py          # Image generation script
├── evaluate.py                 # Evaluation script (FID)
└── README.md
```

## Architecture

### DCGAN Generator
```
z (100-dim) → ConvTranspose2d → 4×4×512 → 8×8×256 → 16×16×128 → 32×32×64 → 64×64×3
              +BatchNorm         +BN         +BN         +BN
              +ReLU              +ReLU       +ReLU       +ReLU      +Tanh
```

### DCGAN Discriminator (with Spectral Normalization)
```
64×64×3 → Conv2d → 32×32×64 → 16×16×128 → 8×8×256 → 4×4×512 → 1×1
          LeakyReLU  +BN        +BN         +BN        +BN
          +SpecNorm  +LeakyReLU  +LeakyReLU  +LeakyReLU  +Sigmoid
```

## Minor Modification: Spectral Normalization

**What:** Every `Conv2d` layer in the Discriminator is wrapped with `nn.utils.spectral_norm()`.

**Why:** Constrains the discriminator's Lipschitz constant, preventing it from becoming too powerful too quickly — the primary cause of vanishing generator gradients and mode collapse.

**Benefit:** More stable training, better gradient flow, no extra hyperparameters.

## Hyperparameters

| Parameter | Value |
|---|---|
| Image size | 64×64 |
| Batch size | 64 |
| Latent dim (z) | 100 |
| Epochs | 50 |
| Learning rate | 0.0002 |
| Adam β₁ | 0.5 |
| Adam β₂ | 0.999 |
| Loss | BCELoss |
| Seed | 42 |

## Results

| Metric | Value |
|---|---|
| FID Score | 105.45 |
| Final G Loss | 3.4675 |
| Final D Loss | 0.3483 |
| Generated images | 15 |

## Dependencies

- PyTorch ≥ 2.0
- torchvision
- NumPy
- SciPy
- PIL (Pillow)

All available in the `dl_assignment` conda environment.
