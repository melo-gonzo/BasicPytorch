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
        print("Validation Finished\n")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.gamma)
        return [optimizer], [lr_scheduler]

    def on_fit_end(self):
        print("\nFinished")
