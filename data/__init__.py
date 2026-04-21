from .acdc_dataset import ACDCSliceDataset, ACDCVolumeDataset
from .synapse_dataset import SynapseSliceDataset, SynapseVolumeDataset


def build_datasets(cfg):
    name = cfg.data.dataset
    if name == "acdc":
        train = ACDCSliceDataset(cfg, split="train", augment=True)
        val = ACDCSliceDataset(cfg, split="valid", augment=False)
        test = ACDCVolumeDataset(cfg, split="test")
        return train, val, test
    if name == "synapse":
        train = SynapseSliceDataset(cfg, split="train", augment=True)
        val = SynapseVolumeDataset(cfg, split="test")  # no official val split; use test volumes for val
        test = SynapseVolumeDataset(cfg, split="test")
        return train, val, test
    raise ValueError(f"unknown dataset: {name}")
