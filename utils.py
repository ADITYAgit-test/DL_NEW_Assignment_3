"""
Utility functions for the DCGAN pipeline.
"""

import os
import random
import json
import logging

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image

import config


def set_seed(seed=config.SEED):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def setup_logging(log_file=config.LOG_FILE):
    """Configure logging to both file and console."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("DCGAN")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def save_checkpoint(gen_model, disc_model, gen_optimizer, disc_optimizer, epoch, loss_dict, path):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "generator_state": gen_model.state_dict(),
        "discriminator_state": disc_model.state_dict(),
        "gen_optimizer_state": gen_optimizer.state_dict(),
        "disc_optimizer_state": disc_optimizer.state_dict(),
        "loss_dict": loss_dict,
    }
    torch.save(checkpoint, path)


def load_checkpoint(gen_model, disc_model, gen_optimizer, disc_optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    gen_model.load_state_dict(checkpoint["generator_state"])
    disc_model.load_state_dict(checkpoint["discriminator_state"])
    gen_optimizer.load_state_dict(checkpoint["gen_optimizer_state"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state"])
    return checkpoint["epoch"], checkpoint["loss_dict"]


def save_generated_images(images, epoch, save_dir=config.OUTPUTS_DIR):
    """Save a grid of generated images during training."""
    os.makedirs(save_dir, exist_ok=True)
    grid = vutils.make_grid(images, nrow=4, normalize=True, padding=2)
    grid_np = grid.cpu().permute(1, 2, 0).numpy()
    img = Image.fromarray((grid_np * 255).astype(np.uint8))
    out_path = os.path.join(save_dir, f"generated_epoch_{epoch:03d}.png")
    img.save(out_path)
    return out_path


def save_individual_samples(fake_images, save_dir=config.SAMPLES_DIR):
    """Save individual generated images with proper naming."""
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i in range(fake_images.size(0)):
        img = fake_images[i].cpu().permute(1, 2, 0).numpy()
        # Denormalize from [-1, 1] to [0, 255]
        img = ((img + 1) / 2 * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        out_path = os.path.join(save_dir, f"{config.SAMPLE_PREFIX}_{i + 1}.png")
        pil_img.save(out_path)
        paths.append(out_path)
    return paths


def save_metrics(metrics, path=config.METRICS_FILE):
    """Save evaluation metrics to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")


def load_metrics(path=config.METRICS_FILE):
    """Load evaluation metrics from JSON."""
    with open(path, "r") as f:
        return json.load(f)
