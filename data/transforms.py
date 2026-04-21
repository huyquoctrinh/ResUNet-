import random

import numpy as np
import torch
import torch.nn.functional as F


def normalize_slice(img: np.ndarray) -> np.ndarray:
    """Zero-mean unit-std per slice; clip extreme values."""
    img = img.astype(np.float32)
    mean = img.mean()
    std = img.std()
    if std < 1e-8:
        return img - mean
    return (img - mean) / std


def resize_pair(img: np.ndarray, label: np.ndarray, size):
    """Resize image (bilinear) and label (nearest) to size=(H, W)."""
    h, w = size
    img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)     # (1,1,H,W)
    lbl_t = torch.from_numpy(label).long().unsqueeze(0).unsqueeze(0).float()
    img_t = F.interpolate(img_t, size=(h, w), mode="bilinear", align_corners=False)
    lbl_t = F.interpolate(lbl_t, size=(h, w), mode="nearest")
    return img_t[0, 0].numpy(), lbl_t[0, 0].long().numpy()


def random_flip2d(img: np.ndarray, label: np.ndarray):
    if random.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        label = np.ascontiguousarray(label[:, ::-1])
    if random.random() < 0.5:
        img = np.ascontiguousarray(img[::-1, :])
        label = np.ascontiguousarray(label[::-1, :])
    return img, label


def random_rotate2d(img: np.ndarray, label: np.ndarray, max_deg: float = 15.0):
    from scipy.ndimage import rotate
    angle = random.uniform(-max_deg, max_deg)
    img = rotate(img, angle, order=1, reshape=False, mode="nearest")
    label = rotate(label, angle, order=0, reshape=False, mode="nearest")
    return img, label


def random_intensity_shift(img: np.ndarray, scale: float = 0.1):
    shift = random.uniform(-scale, scale)
    factor = 1.0 + random.uniform(-scale, scale)
    return img * factor + shift


def augment_2d(img: np.ndarray, label: np.ndarray) -> tuple:
    img, label = random_flip2d(img, label)
    img, label = random_rotate2d(img, label)
    img = random_intensity_shift(img)
    return img, label
