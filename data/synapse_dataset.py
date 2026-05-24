import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


# PraNet-V2 / MIST 9-class ordering (BTCV labels kept in place except:
# 5,9,10,12,13 → 0 (background) and 11 → 5 (Pancreas).
# Result: [BG, Spleen, RK, LK, GB, Pancreas, Liver, Stomach, Aorta].
BTCV_DROP = (5, 9, 10, 12, 13)
BTCV_MERGE_PANCREAS = (11, 5)


def _build_remap_table(mode):
    if mode in (None, "none", "identity"):
        return np.arange(256, dtype=np.int64)
    if mode == "pranet_9class":
        lut = np.arange(256, dtype=np.int64)
        for c in BTCV_DROP:
            lut[c] = 0
        lut[BTCV_MERGE_PANCREAS[0]] = BTCV_MERGE_PANCREAS[1]
        # any raw label >= 14 should map to 0 (safety)
        for c in range(14, 256):
            lut[c] = 0
        return lut
    raise ValueError(f"unknown label_remap: {mode}")


def _rot90_flip(image, label):
    k = random.randint(0, 3)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = random.randint(0, 1)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def _rotate(image, label, max_deg=20.0):
    angle = random.uniform(-max_deg, max_deg)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def _augment(image, label):
    if random.random() > 0.5:
        image, label = _rot90_flip(image, label)
    elif random.random() > 0.5:
        image, label = _rotate(image, label)
    return image, label


class SynapseSliceDataset(Dataset):
    """2D slice dataset for Synapse. Matches PraNet-V2 MIST recipe:
    no per-slice normalization, no empty-slice filter, rot90+flip aug.
    """

    def __init__(self, cfg, split: str = "train", augment: bool = False):
        self.cfg = cfg
        self.root = Path(cfg.data.root)
        self.split = split
        self.augment = augment
        self.img_size = tuple(cfg.data.img_size)
        self.remap = _build_remap_table(
            getattr(cfg.data, "label_remap", None)
        )

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
        image = d["image"].astype(np.float32)            # already in [0, 1]
        label = self.remap[d["label"].astype(np.int64)]

        if self.augment:
            image, label = _augment(image, label)

        h, w = image.shape
        th, tw = self.img_size
        if h != th or w != tw:
            image = zoom(image, (th / h, tw / w), order=3)
            label = zoom(label, (th / h, tw / w), order=0)

        img_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label_t = torch.from_numpy(label.astype(np.int64)).long()
        return img_t, label_t, str(path.name)


class SynapseVolumeDataset(Dataset):
    """3D volume dataset for Synapse test (h5). No normalization — raw image."""

    def __init__(self, cfg, split: str = "test"):
        self.cfg = cfg
        self.root = Path(cfg.data.root)
        self.img_size = tuple(cfg.data.img_size)
        self.remap = _build_remap_table(
            getattr(cfg.data, "label_remap", None)
        )
        vol_dir = self.root / "test_vol_h5_new"
        self.files = sorted(
            [vol_dir / n for n in os.listdir(vol_dir) if n.endswith(".h5")]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with h5py.File(path, "r") as f:
            image = f["image"][...].astype(np.float32)               # (D, H, W)
            label = self.remap[f["label"][...].astype(np.int64)]     # (D, H, W)

        img_t = torch.from_numpy(image).float().unsqueeze(0)   # (1, D, H, W)
        label_t = torch.from_numpy(label).long()               # (D, H, W)
        return img_t, label_t, str(path.name)
