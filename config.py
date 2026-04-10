"""
Configuration file for DCGAN Face Image Generation.
All hyperparameters, paths, and constants in one place.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "image")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
METRICS_DIR = os.path.join(PROJECT_ROOT, "metrics")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")

# ── Reproducibility ────────────────────────────────────────────────────
SEED = 42

# ── Data ───────────────────────────────────────────────────────────────
IMAGE_SIZE = 64          # Resize target (original is 1024x1024)
IMAGE_CHANNELS = 3
BATCH_SIZE = 64          # Falls back to 32 if OOM
NUM_WORKERS = 4

# ── Model ──────────────────────────────────────────────────────────────
Z_DIM = 100              # Latent vector dimension
GEN_FEATURES = 64        # Base feature maps for Generator
DISC_FEATURES = 64       # Base feature maps for Discriminator

# ── Training ───────────────────────────────────────────────────────────
EPOCHS = 50
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# ── Minor Modification: Spectral Normalization ─────────────────────────
# Applied to all Conv2d layers in the Discriminator.
# This constrains the Lipschitz constant of the discriminator,
# preventing it from becoming too strong and stabilizing GAN training.
USE_SPECTRAL_NORM = True

# ── Checkpointing & Logging ───────────────────────────────────────────
SAVE_INTERVAL = 5        # Save checkpoint & samples every N epochs
LOG_FILE = os.path.join(LOGS_DIR, "train.log")
BEST_MODEL_G = os.path.join(MODELS_DIR, "generator_best.pth")
BEST_MODEL_D = os.path.join(MODELS_DIR, "discriminator_best.pth")

# ── Generation ─────────────────────────────────────────────────────────
NUM_SAMPLES = 15
SAMPLE_PREFIX = "sample"

# ── Evaluation ─────────────────────────────────────────────────────────
METRICS_FILE = os.path.join(METRICS_DIR, "metrics.json")
GENERATED_GRID = os.path.join(OUTPUTS_DIR, "generated_grid.png")

# ── Report ─────────────────────────────────────────────────────────────
REPORT_FILE = os.path.join(REPORT_DIR, "assignment_report.txt")
