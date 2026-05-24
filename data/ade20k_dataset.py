import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ADE20K_LUT_SIZE = 4096


def _decode_ade20k_mask(seg_rgb: np.ndarray) -> np.ndarray:
    """ADE20K RGB encoding: class_id = (R // 10) * 256 + G."""
    return (seg_rgb[:, :, 0].astype(np.int32) // 10) * 256 + seg_rgb[:, :, 1].astype(np.int32)


def _build_class_mapping(data_dir: str, num_top_classes: int, cache_path: Path) -> dict:
    """Scan training set to find top-N classes by pixel frequency. Cached."""
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    from datasets import load_dataset

    ds = load_dataset("parquet", data_dir=data_dir, split="train")

    pixel_counts: Counter = Counter()
    id_to_name: dict = {}

    print(f"[ADE20K] scanning {len(ds)} training samples to build class mapping ...")
    for i in range(len(ds)):
        row = ds[i]
        seg_np = np.array(row["segmentations"][0])
        class_map = _decode_ade20k_mask(seg_np)
        unique, counts = np.unique(class_map, return_counts=True)
        for cls_id, cnt in zip(unique.tolist(), counts.tolist()):
            if cls_id > 0:
                pixel_counts[cls_id] += cnt

        objects = row["objects"]
        if isinstance(objects, dict):
            for ndx, name in zip(objects.get("name_ndx", []), objects.get("name", [])):
                ndx = int(ndx)
                if ndx not in id_to_name:
                    id_to_name[ndx] = name
        elif isinstance(objects, list):
            for obj in objects:
                ndx = int(obj["name_ndx"])
                if ndx not in id_to_name:
                    id_to_name[ndx] = obj["name"]

        if (i + 1) % 5000 == 0:
            print(f"  scanned {i + 1}/{len(ds)}")

    top_classes = sorted(
        [cls_id for cls_id, _ in pixel_counts.most_common(num_top_classes)]
    )

    mapping = {"0": 0}
    names = ["void"]
    for i, cls_id in enumerate(top_classes):
        mapping[str(cls_id)] = i + 1
        names.append(id_to_name.get(cls_id, f"class_{cls_id}"))

    result = {
        "mapping": mapping,
        "class_names": names,
        "num_classes": len(names),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ADE20K] class mapping saved to {cache_path} ({len(names)} classes)")

    return result


class ADE20KDataset(Dataset):
    """ADE20K semantic segmentation dataset loaded from HuggingFace parquet files.

    On first use, scans training data to identify the top-N most frequent
    classes (by pixel count) and caches a contiguous remapping as JSON.
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, cfg, split: str = "train", augment: bool = False):
        from datasets import load_dataset

        self.cfg = cfg
        self.augment = augment
        self.img_size = tuple(cfg.data.img_size)
        root = Path(cfg.data.root)

        data_dir = str(root / "data") if (root / "data").is_dir() else str(root)

        split_name = "validation" if split in ("val", "test", "valid", "validation") else split
        self.ds = load_dataset("parquet", data_dir=data_dir, split=split_name)

        num_top = int(getattr(cfg.data, "num_top_classes", 150))
        cache_path = root / f"class_mapping_{num_top}.json"
        mapping_info = _build_class_mapping(data_dir, num_top, cache_path)

        self.class_names = mapping_info["class_names"]
        self.num_mapped_classes = mapping_info["num_classes"]

        self.remap_lut = np.zeros(ADE20K_LUT_SIZE, dtype=np.int64)
        for k, v in mapping_info["mapping"].items():
            idx = int(k)
            if idx < ADE20K_LUT_SIZE:
                self.remap_lut[idx] = v

    def __len__(self):
        return len(self.ds)

    def _random_scale_crop(self, img: np.ndarray, label: np.ndarray):
        scale = random.uniform(0.5, 2.0)
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        lbl_t = torch.from_numpy(label.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        img_t = F.interpolate(img_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        lbl_t = F.interpolate(lbl_t, size=(new_h, new_w), mode="nearest")
        img = img_t[0].permute(1, 2, 0).numpy()
        label = lbl_t[0, 0].long().numpy()

        th, tw = self.img_size
        if new_h < th or new_w < tw:
            pad_h = max(th - new_h, 0)
            pad_w = max(tw - new_w, 0)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode="constant")
            new_h, new_w = img.shape[0], img.shape[1]

        y = random.randint(0, new_h - th)
        x = random.randint(0, new_w - tw)
        return img[y : y + th, x : x + tw], label[y : y + th, x : x + tw]

    def _augment_pair(self, img: np.ndarray, label: np.ndarray):
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            label = np.ascontiguousarray(label[:, ::-1])
        img, label = self._random_scale_crop(img, label)
        return img, label

    def __getitem__(self, idx):
        row = self.ds[idx]

        img_pil = row["image"]
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")
        img = np.array(img_pil, dtype=np.float32) / 255.0

        seg_np = np.array(row["segmentations"][0])
        class_map = _decode_ade20k_mask(seg_np)
        label = np.where(
            class_map < ADE20K_LUT_SIZE, self.remap_lut[class_map], 0
        ).astype(np.int64)

        if self.augment:
            img, label = self._augment_pair(img, label)

        h, w = img.shape[:2]
        th, tw = self.img_size
        if h != th or w != tw:
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
            lbl_t = torch.from_numpy(label.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            img_t = F.interpolate(img_t, size=(th, tw), mode="bilinear", align_corners=False)
            lbl_t = F.interpolate(lbl_t, size=(th, tw), mode="nearest")
            img = img_t[0].permute(1, 2, 0).numpy()
            label = lbl_t[0, 0].long().numpy()

        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD

        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        label_t = torch.from_numpy(label.copy()).long()
        filename = row.get("filename", f"sample_{idx}")

        return img_t, label_t, filename
