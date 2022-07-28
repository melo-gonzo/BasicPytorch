from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, size, length):
        super().__init__()
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class RandomDataModule(pl.LightningDataModule):
    def __init__(self, size: int = 32, length: int = 256, batch_size: int = 2):
        super().__init__()
        self.size = size
        self.length = length
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        """
        Args:
            stage: used to separate setup logic for trainer.{fit,validate,test}.
                If setup is called with stage = None, we assume all stages have been set-up.
        """
        # if stage in (None, "fit"):
        self.train_dataset = RandomDataset(self.size, self.length)

        # if stage in (None, "test"):
        self.test_dataset = RandomDataset(self.size, self.length)

        # if stage in (None, "val"):
        self.val_dataset = RandomDataset(self.size, self.length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
