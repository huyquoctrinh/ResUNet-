import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PolypDataset(Dataset):
    """Training dataset for polyp segmentation with seed-synchronized augmentation."""

    def __init__(self, root, img_size=352):
        self.img_dir = os.path.join(root, 'image')
        self.mask_dir = os.path.join(root, 'masks')
        self.filenames = sorted(os.listdir(self.img_dir))
        self.img_size = img_size

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert('L')

        seed = random.randint(0, 2**32 - 1)

        random.seed(seed)
        torch.manual_seed(seed)
        img = self._augment_image(img)

        random.seed(seed)
        torch.manual_seed(seed)
        mask = self._augment_mask(mask)

        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask[0] > 0.5).long()

        return {'image': img, 'label': mask, 'case_name': fname}

    def _augment_image(self, img):
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
        if random.random() > 0.5:
            img = transforms.functional.vflip(img)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            img = transforms.functional.rotate(img, angle)
        return img

    def _augment_mask(self, mask):
        if random.random() > 0.5:
            mask = transforms.functional.hflip(mask)
        if random.random() > 0.5:
            mask = transforms.functional.vflip(mask)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            mask = transforms.functional.rotate(mask, angle, interpolation=Image.NEAREST)
        return mask


class PolypTestDataset(Dataset):
    """Test dataset for polyp segmentation (no augmentation).

    When ``original_resolution=True`` (default, for benchmarking), the label is
    returned at its original size and ``orig_size`` is included so the caller
    can resize predictions back before computing metrics.

    When ``original_resolution=False`` (for training-time validation), the label
    is resized to ``img_size`` so it matches model output directly.
    """

    def __init__(self, root, img_size=352, original_resolution=True):
        self.img_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'masks')
        self.filenames = sorted(os.listdir(self.img_dir))
        self.img_size = img_size
        self.original_resolution = original_resolution

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert('L')

        img = self.img_transform(img)

        if self.original_resolution:
            mask_np = np.array(mask)
            orig_h, orig_w = mask_np.shape
            label = torch.from_numpy((mask_np > 127).astype(np.int64)).long()
            return {
                'image': img,
                'label': label,
                'orig_size': torch.tensor([orig_h, orig_w]),
                'case_name': fname,
            }
        else:
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            label = (np.array(mask) > 127).astype(np.int64)
            return {
                'image': img,
                'label': torch.from_numpy(label).long(),
                'case_name': fname,
            }
