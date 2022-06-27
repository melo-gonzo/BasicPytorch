import operator
import time
import traceback

# from argparse import ArgumentParser
from functools import reduce
from typing import Dict, Optional, Union
from types import SimpleNamespace
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningCLI
from torch.utils.data import DataLoader, Dataset
import torchmetrics

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


class BoringModel(LightningModule):
    def __init__(
        self,
        channels: int = 32,
        hidden_dim: int = 64,
        depth: int = 2,
        activation: str = "ReLU"
    ):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.activation = getattr(torch.nn, activation)()
        self.model = []
        self.model.append(torch.nn.Linear(channels, hidden_dim))
        self.model.append(self.activation)
        for _ in range(self.depth):
            self.model.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.model.append(self.activation)
        self.model.append(torch.nn.Linear(hidden_dim, 2))
        self.model = torch.nn.Sequential(*self.model)
        self.lr = 0.001
        self.gamma = 0.1
        # self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        metric = self(batch).sum()
        return {"metric": metric}

    def test_epoch_end(self, outputs):
        list_of_metrics = [output["metric"] for output in outputs]
        avg_metric = torch.stack(list_of_metrics).mean()
        self.log("metric", avg_metric, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("val_loss", loss, prog_bar=True)

    def validation_epoch_end(self, outputs):
        print("validation finished")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.gamma)
        return [optimizer], [lr_scheduler]

    def on_fit_end(self):
        print("\nFinished")


def get_from_dict(input_dict, map_list):
    return reduce(operator.getitem, map_list, input_dict)


def set_in_dict(input_dict, map_list, value):
    get_from_dict(input_dict, map_list[:-1])[map_list[-1]] = value


if __name__ == "__main__":
    cli = LightningCLI(
        BoringModel,
        RandomDataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None},
    )
