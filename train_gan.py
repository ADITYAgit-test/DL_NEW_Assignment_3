"""
DCGAN Training Script — Realistic Face Image Generation

Minor Modification: Spectral Normalization on Discriminator
────────────────────────────────────────────────────────────
What changed:
    Every Conv2d layer in the Discriminator is wrapped with
    torch.nn.utils.spectral_norm().

Why it helps:
    Spectral normalization constrains the Lipschitz constant of the
    discriminator by dividing the weight matrix by its largest singular
    value. This prevents the discriminator from becoming too powerful
    too quickly — a common cause of vanishing generator gradients.

How it improves stability:
    • Smoother loss curves for both G and D
    • Reduced risk of mode collapse
    • Better gradient signal to the generator
    • No extra hyperparameters to tune
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import config
from utils import (
    set_seed,
    get_device,
    setup_logging,
    save_checkpoint,
    save_generated_images,
)


# ─────────────────────────── Dataset ────────────────────────────────────

class FaceDataset(Dataset):
    """Loads face images from the data/image directory."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# ─────────────────────────── Generator ──────────────────────────────────

class Generator(nn.Module):
    """
    DCGAN Generator Architecture.

    Input:  z ∈ R^z_dim  (latent vector)
    Output: image ∈ R^(3 x 64 x 64)
    """

    def __init__(self, z_dim, gen_features, img_channels):
        super().__init__()
        self.network = nn.Sequential(
            # Block 1: z -> 4x4
            nn.ConvTranspose2d(z_dim, gen_features * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(gen_features * 8),
            nn.ReLU(True),

            # Block 2: 4x4 -> 8x8
            nn.ConvTranspose2d(gen_features * 8, gen_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gen_features * 4),
            nn.ReLU(True),

            # Block 3: 8x8 -> 16x16
            nn.ConvTranspose2d(gen_features * 4, gen_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gen_features * 2),
            nn.ReLU(True),

            # Block 4: 16x16 -> 32x32
            nn.ConvTranspose2d(gen_features * 2, gen_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gen_features),
            nn.ReLU(True),

            # Block 5: 32x32 -> 64x64
            nn.ConvTranspose2d(gen_features, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


# ─────────────────────────── Discriminator ──────────────────────────────

class Discriminator(nn.Module):
    """
    DCGAN Discriminator Architecture.

    MINOR MODIFICATION: Spectral Normalization applied to all Conv2d layers.

    Input:  image ∈ R^(3 x 64 x 64)
    Output: scalar probability ∈ [0, 1]
    """

    def __init__(self, img_channels, disc_features):
        super().__init__()

        def spectral_norm_wrapper(layer):
            """Apply spectral normalization if enabled."""
            if config.USE_SPECTRAL_NORM:
                return nn.utils.spectral_norm(layer)
            return layer

        self.network = nn.Sequential(
            # Block 1: 64x64 -> 32x32  (no BatchNorm as per DCGAN paper)
            spectral_norm_wrapper(nn.Conv2d(img_channels, disc_features, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: 32x32 -> 16x16
            spectral_norm_wrapper(nn.Conv2d(disc_features, disc_features * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: 16x16 -> 8x8
            spectral_norm_wrapper(nn.Conv2d(disc_features * 2, disc_features * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: 8x8 -> 4x4
            spectral_norm_wrapper(nn.Conv2d(disc_features * 4, disc_features * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output: 4x4 -> 1x1
            spectral_norm_wrapper(nn.Conv2d(disc_features * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x).view(-1, 1).squeeze(1)


# ─────────────────────────── Training ───────────────────────────────────

def train():
    # Reproducibility
    set_seed()

    # Device
    device = get_device()

    # Logger
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("DCGAN Training Started")
    logger.info(f"Device: {device}")
    logger.info(f"Spectral Normalization: {config.USE_SPECTRAL_NORM}")
    logger.info(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Latent dim (z_dim): {config.Z_DIM}")
    logger.info("=" * 60)

    # ── Data ───────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
    ])

    dataset = FaceDataset(config.DATA_DIR, transform=transform)
    logger.info(f"Dataset size: {len(dataset)} images")

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
    except RuntimeError:
        # Fallback if too many workers
        logger.warning("Falling back to num_workers=0")
        dataloader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    # ── Models ─────────────────────────────────────────────────────
    gen = Generator(config.Z_DIM, config.GEN_FEATURES, config.IMAGE_CHANNELS).to(device)
    disc = Discriminator(config.IMAGE_CHANNELS, config.DISC_FEATURES).to(device)

    logger.info(f"Generator parameters: {sum(p.numel() for p in gen.parameters()):,}")
    logger.info(f"Discriminator parameters: {sum(p.numel() for p in disc.parameters()):,}")

    # ── Loss & Optimizers ──────────────────────────────────────────
    criterion = nn.BCELoss()
    gen_optimizer = optim.Adam(gen.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    disc_optimizer = optim.Adam(disc.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))

    # Fixed noise for consistent sample visualization across epochs
    fixed_noise = torch.randn(16, config.Z_DIM, 1, 1, device=device)

    # ── Training Loop ──────────────────────────────────────────────
    gen_losses = []
    disc_losses = []
    best_fid = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()

        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        n_batches = 0

        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # ── Labels ─────────────────────────────────────────────
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # ── Train Discriminator ────────────────────────────────
            disc_optimizer.zero_grad()

            # Real images
            disc_real = disc(real_images)
            d_loss_real = criterion(disc_real, real_labels)

            # Fake images
            noise = torch.randn(batch_size, config.Z_DIM, 1, 1, device=device)
            fake_images = gen(noise)
            disc_fake = disc(fake_images.detach())
            d_loss_fake = criterion(disc_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            disc_optimizer.step()

            # ── Train Generator ────────────────────────────────────
            gen_optimizer.zero_grad()

            output = disc(fake_images)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            gen_optimizer.step()

            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()
            n_batches += 1

        # ── Epoch Summary ──────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        avg_g_loss = g_epoch_loss / n_batches
        avg_d_loss = d_epoch_loss / n_batches

        gen_losses.append(avg_g_loss)
        disc_losses.append(avg_d_loss)

        msg = (
            f"Epoch [{epoch}/{config.EPOCHS}] "
            f"Time: {epoch_time:.1f}s "
            f"G_Loss: {avg_g_loss:.4f} "
            f"D_Loss: {avg_d_loss:.4f}"
        )
        logger.info(msg)

        # ── Save Checkpoint & Samples Every N Epochs ───────────────
        if epoch % config.SAVE_INTERVAL == 0 or epoch == config.EPOCHS:
            # Save checkpoint
            ckpt_path = os.path.join(config.MODELS_DIR, f"checkpoint_epoch_{epoch:03d}.pth")
            save_checkpoint(
                gen, disc, gen_optimizer, disc_optimizer,
                epoch,
                {"gen_loss": avg_g_loss, "disc_loss": avg_d_loss},
                ckpt_path,
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

            # Save generated sample grid
            gen.eval()
            with torch.no_grad():
                fake_samples = gen(fixed_noise)
            grid_path = save_generated_images(fake_samples, epoch)
            gen.train()
            logger.info(f"Sample grid saved: {grid_path}")

            # Track best model
            if avg_g_loss < best_fid:
                best_fid = avg_g_loss
                torch.save(gen.state_dict(), config.BEST_MODEL_G)
                torch.save(disc.state_dict(), config.BEST_MODEL_D)
                logger.info("New best model saved.")

    # ── Final ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Final Generator Loss: {gen_losses[-1]:.4f}")
    logger.info(f"Final Discriminator Loss: {disc_losses[-1]:.4f}")
    logger.info(f"Best model saved to: {config.BEST_MODEL_G}")
    logger.info("=" * 60)

    # Save loss history for evaluation
    loss_data = {
        "gen_losses": gen_losses,
        "disc_losses": disc_losses,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "spectral_norm": config.USE_SPECTRAL_NORM,
    }
    loss_path = os.path.join(config.LOGS_DIR, "loss_history.json")
    import json
    with open(loss_path, "w") as f:
        json.dump(loss_data, f, indent=2)
    logger.info(f"Loss history saved: {loss_path}")


if __name__ == "__main__":
    train()
