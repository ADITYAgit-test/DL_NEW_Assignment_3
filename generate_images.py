"""
Generate synthetic face images from trained DCGAN.

Loads the best checkpoint and generates 15 individual sample images
plus a visualization grid.
"""

import os
import torch

import config
from utils import set_seed, get_device, save_individual_samples
from train_gan import Generator


def generate():
    set_seed()
    device = get_device()

    # Load generator
    gen = Generator(config.Z_DIM, config.GEN_FEATURES, config.IMAGE_CHANNELS).to(device)

    # Try best model first, fallback to latest checkpoint
    if os.path.exists(config.BEST_MODEL_G):
        gen.load_state_dict(torch.load(config.BEST_MODEL_G, map_location=device))
        print(f"Loaded best model from {config.BEST_MODEL_G}")
    else:
        # Find latest checkpoint
        ckpts = sorted(
            [f for f in os.listdir(config.MODELS_DIR) if f.startswith("checkpoint_")]
        )
        if ckpts:
            latest = os.path.join(config.MODELS_DIR, ckpts[-1])
            ckpt = torch.load(latest, map_location=device)
            gen.load_state_dict(ckpt["generator_state"])
            print(f"Loaded checkpoint from {latest} (epoch {ckpt['epoch']})")
        else:
            print("ERROR: No trained model found. Run train_gan.py first.")
            return

    gen.eval()

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(config.NUM_SAMPLES, config.Z_DIM, 1, 1, device=device)
        fake_images = gen(noise)

    # Save individual images: sample_1.png ... sample_15.png
    paths = save_individual_samples(fake_images)
    print(f"\nGenerated {len(paths)} images:")
    for p in paths:
        print(f"  {p}")

    # Also save a grid for visualization
    import torchvision.utils as vutils
    from PIL import Image
    import numpy as np

    grid = vutils.make_grid(fake_images, nrow=5, normalize=True, padding=4)
    grid_np = grid.cpu().permute(1, 2, 0).numpy()
    img = Image.fromarray((grid_np * 255).astype(np.uint8))
    img.save(config.GENERATED_GRID)
    print(f"\nGrid saved to {config.GENERATED_GRID}")

    print("\nGeneration complete!")


if __name__ == "__main__":
    generate()
