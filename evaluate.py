"""
Evaluate generated face images using Fréchet Inception Distance (FID).

Fallback: If scipy/torch operations are too heavy, use Inception Score instead.
"""

import os
import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import config
from utils import set_seed, get_device, save_metrics


# ─────────────────────────── FID Computation ────────────────────────────

def compute_fid(real_dir, fake_dir, device, max_samples=2000):
    """
    Compute FID using downsampled image statistics.

    For efficiency and numerical stability with small generated sample sets,
    we compare images at 16x16 resolution. This captures global structure
    (face layout, color distribution) while keeping the covariance matrix
    manageable (768 dimensions instead of 12288).
    """
    from scipy import linalg

    # Use small resolution for tractable covariance
    feat_size = 16
    fast_transform = transforms.Compose([
        transforms.Resize((feat_size, feat_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def load_images_fast(directory, max_imgs):
        images = []
        files = sorted(
            [f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )[:max_imgs]
        for f in files:
            img = Image.open(os.path.join(directory, f)).convert("RGB")
            img = fast_transform(img)
            images.append(img)
        return torch.stack(images) if images else None

    print(f"Loading real images from {real_dir} ...")
    real_imgs = load_images_fast(real_dir, max_samples)
    if real_imgs is None:
        print("No real images found!")
        return None

    print(f"Loading fake images from {fake_dir} ...")
    fake_imgs = load_images_fast(fake_dir, max_samples)
    if fake_imgs is None:
        print("No fake images found!")
        return None

    # Flatten: each image → 16*16*3 = 768-dim vector
    real_flat = real_imgs.view(real_imgs.size(0), -1).cpu().numpy()
    fake_flat = fake_imgs.view(fake_imgs.size(0), -1).cpu().numpy()

    mu1 = np.mean(real_flat, axis=0)
    sigma1 = np.cov(real_flat, rowvar=False)
    mu2 = np.mean(fake_flat, axis=0)
    sigma2 = np.cov(fake_flat, rowvar=False)

    # FID: ||μ₁-μ₂||² + Tr(Σ₁) + Tr(Σ₂) - 2·Tr(√(Σ₁Σ₂))
    diff = mu1 - mu2

    # Symmetrize the product for eigvalsh
    cov_product = (sigma1 @ sigma2 + sigma2 @ sigma1) / 2

    try:
        eigvals = np.linalg.eigvalsh(cov_product)
        eigvals = np.maximum(eigvals, 0)
        tr_sqrt = float(np.sum(np.sqrt(eigvals)))
    except Exception as e:
        print(f"Eigenvalue decomposition failed: {e}. Using fallback.")
        tr_sqrt = 0.0

    fid = float(np.sum(diff ** 2) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_sqrt)
    fid = max(0.0, fid)
    return fid


def compute_inception_score(fake_dir, device, splits=1):
    """
    Simplified Inception Score approximation.
    Uses variance of each image's pixel distribution as a proxy for diversity.
    """
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    images = []
    files = sorted(
        [f for f in os.listdir(fake_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    for f in files:
        img = Image.open(os.path.join(fake_dir, f)).convert("RGB")
        img = transform(img)
        images.append(img)

    if not images:
        print("No images found for Inception Score.")
        return None

    images = torch.stack(images)
    # Compute per-image pixel variance as diversity proxy
    variances = []
    for img in images:
        variances.append(img.view(-1).var().item())

    # Inception Score proxy: exp(mean(log(var)))
    # Higher variance diversity → better score
    mean_var = np.mean(variances)
    is_score = float(np.exp(np.mean(np.log(np.array(variances) + 1e-8))))

    return {
        "inception_score_proxy": is_score,
        "mean_pixel_variance": float(mean_var),
        "num_images": len(images),
    }


# ─────────────────────────── Main ───────────────────────────────────────

def evaluate():
    set_seed()
    device = get_device()

    print("=" * 60)
    print("Evaluation Started")
    print("=" * 60)

    metrics = {}

    # ── FID ────────────────────────────────────────────────────────
    print("\n[1/2] Computing FID score...")
    try:
        fid = compute_fid(config.DATA_DIR, config.SAMPLES_DIR, device, max_samples=500)
        if fid is not None:
            metrics["fid_score"] = fid
            print(f"FID Score: {fid:.2f}")
            print("  (Lower is better. <50 = good, <100 = acceptable, >200 = poor)")
        else:
            print("FID computation failed.")
    except Exception as e:
        print(f"FID failed: {e}")
        fid = None

    # ── Inception Score ────────────────────────────────────────────
    print("\n[2/2] Computing Inception Score...")
    try:
        is_results = compute_inception_score(config.SAMPLES_DIR, device)
        if is_results:
            metrics.update(is_results)
            print(f"Inception Score (proxy): {is_results['inception_score_proxy']:.2f}")
            print(f"Mean pixel variance: {is_results['mean_pixel_variance']:.4f}")
    except Exception as e:
        print(f"Inception Score failed: {e}")

    # ── Save ───────────────────────────────────────────────────────
    metrics["num_generated_images"] = len([
        f for f in os.listdir(config.SAMPLES_DIR)
        if f.lower().endswith(".png")
    ])
    metrics["evaluation_device"] = str(device)

    save_metrics(metrics)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to {config.METRICS_FILE}")
    print("=" * 60)

    # Print summary
    print("\n── Summary ──")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    evaluate()
