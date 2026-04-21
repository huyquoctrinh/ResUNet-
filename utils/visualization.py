import numpy as np
import torch


def label_to_color(label: np.ndarray, num_classes: int) -> np.ndarray:
    """Simple fixed palette for overlay visualization."""
    palette = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255],
            [0, 128, 255],
            [128, 255, 0],
            [255, 128, 128],
            [128, 255, 128],
            [128, 128, 255],
        ],
        dtype=np.uint8,
    )
    if num_classes > len(palette):
        extra = np.random.default_rng(0).integers(0, 255, size=(num_classes - len(palette), 3), dtype=np.uint8)
        palette = np.concatenate([palette, extra], axis=0)
    return palette[label]


def make_grid(img: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    """Return a triplet (img|pred|gt) uint8 image for TB logging."""
    if img.ndim == 4:
        img = img[0]
    img_np = img.detach().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_rgb = np.repeat((img_np[0] * 255).astype(np.uint8)[..., None], 3, axis=-1)
    pred_rgb = label_to_color(pred.detach().cpu().numpy().astype(np.int64), num_classes)
    tgt_rgb = label_to_color(target.detach().cpu().numpy().astype(np.int64), num_classes)
    return np.concatenate([img_rgb, pred_rgb, tgt_rgb], axis=1)
