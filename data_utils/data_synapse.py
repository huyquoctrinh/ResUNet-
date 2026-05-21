import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


SYNAPSE_9CLASS_REMAP = {5: 0, 9: 0, 10: 0, 12: 0, 13: 0, 11: 5}


def remap_labels(label, nclass=9):
    """Remap Synapse labels from 14 classes to 9 classes."""
    if nclass == 9:
        for src, dst in SYNAPSE_9CLASS_REMAP.items():
            label[label == src] = dst
    return label


class Synapse_dataset(Dataset):
    """Synapse multi-organ segmentation dataset.

    Expected layout under ``base_dir``::

        base_dir/
            train.txt
            test.txt
            train_npz_new/    # 2-D slice .npz  (keys: image, label)
            test_vol_h5_new/  # 3-D volume .h5   (keys: image, label)
    """

    def __init__(self, base_dir, list_dir, split, nclass=9, transform=None):
        self.transform = transform
        self.split = split
        self.nclass = nclass
        self.data_dir = base_dir

        if self.split == "train":
            self.sample_list = open(
                os.path.join(list_dir, self.split + '.txt')
            ).readlines()
        else:
            vol_dir = os.path.join(base_dir, 'test_vol_h5_new')
            self.sample_list = sorted([
                f.replace('.npy.h5', '')
                for f in os.listdir(vol_dir)
                if f.endswith('.h5')
            ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            if not slice_name.endswith('.npz'):
                slice_name = slice_name + '.npz'
            data_path = os.path.join(self.data_dir, 'train_npz_new', slice_name)
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx]
            filepath = os.path.join(
                self.data_dir, 'test_vol_h5_new', vol_name + '.npy.h5'
            )
            data = h5py.File(filepath, 'r')
            image, label = data['image'][:], data['label'][:]

        label = remap_labels(label, self.nclass)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n') if self.split == "train" else self.sample_list[idx]
        return sample
