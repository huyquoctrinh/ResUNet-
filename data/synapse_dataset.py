import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import augment_2d, normalize_slice, resize_pair


class SynapseSliceDataset(Dataset):
    """2D slice dataset for Synapse (CASCADE preprocessing). npz with `image`, `label`."""

    def __init__(self, cfg, split: str = "train", augment: bool = False):
        self.cfg = cfg
        self.root = Path(cfg.data.root)
        self.split = split
        self.augment = augment
        self.img_size = tuple(cfg.data.img_size)

        list_file = self.root / f"{split}.txt"
        sub = "train_npz_new" if split == "train" else split
        if list_file.exists():
            with open(list_file) as f:
                names = [line.strip() for line in f if line.strip()]
            names = [n if n.endswith(".npz") else f"{n}.npz" for n in names]
        else:
            names = sorted(os.listdir(self.root / sub))
        self.files = [self.root / sub / n for n in names if n.endswith(".npz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        d = np.load(path)
        img = d["image"].astype(np.float32)
        label = d["label"].astype(np.int64)

        img = normalize_slice(img)
        if img.shape != self.img_size:
            img, label = resize_pair(img, label, self.img_size)
        if self.augment:
            img, label = augment_2d(img, label)

        img = torch.from_numpy(img.copy()).float().unsqueeze(0)
        label = torch.from_numpy(label.copy()).long()
        return img, label, str(path.name)


class SynapseVolumeDataset(Dataset):
    """3D volume dataset for Synapse test. Files are h5 in `test_vol_h5_new/`."""

    def __init__(self, cfg, split: str = "test"):
        self.cfg = cfg
        self.root = Path(cfg.data.root)
        self.img_size = tuple(cfg.data.img_size)
        vol_dir = self.root / "test_vol_h5_new"
        self.files = sorted([vol_dir / n for n in os.listdir(vol_dir) if n.endswith(".h5")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with h5py.File(path, "r") as f:
            img = f["image"][...].astype(np.float32)     # (D, H, W)
            label = f["label"][...].astype(np.int64)     # (D, H, W)

        mean, std = img.mean(), img.std()
        img = (img - mean) / (std + 1e-8)

        img_t = torch.from_numpy(img).float().unsqueeze(0)   # (1, D, H, W)
        label_t = torch.from_numpy(label).long()             # (D, H, W)
        return img_t, label_t, str(path.name)
