"""Shared settings, data loading and model definition.

Place this file (and the other scripts) in the ROOT of the xai-series repo, so that:
    data/brain_mri/training/<class folders>
    data/brain_mri/testing/<class folders>
exist next to it.
"""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# ----------------------------------------------------------------------------
# Paths and hyper-parameters
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "brain_mri"
TRAIN_DIR = DATA_DIR / "training"
TEST_DIR = DATA_DIR / "testing"
MODEL_PATH = ROOT / "models" / "brain_cnn.pt"
OUT_DIR = ROOT / "outputs"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # keep 0 on Windows / Colab to avoid multiprocessing problems
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def get_transforms(train: bool) -> transforms.Compose:
    steps = [transforms.Grayscale(num_output_channels=1),
             transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        steps += [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    return transforms.Compose(steps)


def get_datasets():
    """Returns (train_ds_with_augmentation, train_ds_without_augmentation, test_ds)."""
    for d in (TRAIN_DIR, TEST_DIR):
        if not d.exists():
            raise FileNotFoundError(
                f"{d} not found. Copy the dataset 'Training'/'Testing' folders into "
                f"data/brain_mri/ and rename them to 'training'/'testing' (lowercase)."
            )
    train_aug = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(True))
    train_plain = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(False))
    test_ds = datasets.ImageFolder(TEST_DIR, transform=get_transforms(False))
    return train_aug, train_plain, test_ds


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Undo Normalize([0.5],[0.5]) for display."""
    return (t * 0.5 + 0.5).clamp(0, 1)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
# Captum's LRP only supports certain layer types (Conv2d, Linear, BatchNorm2d,
# MaxPool2d, AdaptiveAvgPool2d, ReLU, Dropout...). Residual connections (ResNet)
# and nn.Flatten are NOT supported, so this is a plain VGG-style CNN where every
# layer is a separate module (no re-used ReLU modules, no in-place ops).
def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2),
    )


class BrainCNN(nn.Module):
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)  # functional call -> not a module, LRP is fine with it
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)

    def gradcam_layer(self) -> nn.Module:
        """ReLU of the last conv block (28x28 feature map) used for Grad-CAM."""
        return self.features[3][2]
