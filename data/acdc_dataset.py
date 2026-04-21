import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import augment_2d, normalize_slice, resize_pair


class ACDCSliceDataset(Dataset):
    """2D slice dataset for ACDC. Reads `.npz` with keys `img`, `label`."""

    def __init__(self, cfg, split: str = "train", augment: bool = False):
        self.cfg = cfg
        self.root = Path(cfg.data.root)
        self.split = split
        self.augment = augment
        self.img_size = tuple(cfg.data.img_size)

        list_file = self.root / "lists_ACDC" / f"{split}.txt"
        if list_file.exists():
            with open(list_file) as f:
                names = [line.strip() for line in f if line.strip()]
        else:
            names = sorted(os.listdir(self.root / split))
        self.files = [self.root / split / n for n in names if n.endswith(".npz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        d = np.load(path)
        img = d["img"]
        label = d["label"].astype(np.int64)

        img = normalize_slice(img)
        if img.shape != self.img_size:
            img, label = resize_pair(img, label, self.img_size)
        if self.augment:
            img, label = augment_2d(img, label)

        img = torch.from_numpy(img.copy()).float().unsqueeze(0)        # (1, H, W)
        label = torch.from_numpy(label.copy()).long()                  # (H, W)
        return img, label, str(path.name)


class ACDCVolumeDataset(Dataset):
    """3D volume dataset for ACDC test. Returns (C,D,H,W) volumes and (D,H,W) labels."""

    def __init__(self, cfg, split: str = "test"):
        self.cfg = cfg
        self.root = Path(cfg.data.root)
        self.split = split
        self.img_size = tuple(cfg.data.img_size)

        list_file = self.root / "lists_ACDC" / f"{split}.txt"
        if list_file.exists():
            with open(list_file) as f:
                names = [line.strip() for line in f if line.strip()]
        else:
            names = sorted(os.listdir(self.root / split))
        self.files = [self.root / split / n for n in names if n.endswith(".npz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        d = np.load(path)
        img = d["img"].astype(np.float32)      # (D, H, W)
        label = d["label"].astype(np.int64)    # (D, H, W)

        # per-volume normalization
        mean, std = img.mean(), img.std()
        img = (img - mean) / (std + 1e-8)

        img_t = torch.from_numpy(img).float().unsqueeze(0)   # (1, D, H, W)
        label_t = torch.from_numpy(label).long()             # (D, H, W)
        return img_t, label_t, str(path.name)
