import matplotlib.pyplot as plt
import os
import time

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch import nn
from torch.utils.data import DataLoader, random_split
# from torchmetrics.functional import accuracy
import torchmetrics
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from pytorch_lightning.utilities.cli import LightningCLI
import yaml


class Basic2DCNN(LightningModule):
    def __init__(
        self, channels, width, height, num_classes, hidden_size=64, learning_rate=2e-4
    ):
        super().__init__()
        # We take in input dimensions as parameters and use those to dynamically build model.
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.loss = nn.CrossEntropyLoss()
        L1_filters = 32
        L2_filters = 64
        # TODO: fix scale hack
        SCALE=3.5

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, L1_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(L1_filters, L2_filters,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(L2_filters * int(280/SCALE - 1) * int(280/SCALE - 1), 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        accuracy = self.train_accuracy(logits, y)
        self.log("train_acc", accuracy, prog_bar=True)
        # n = 2
        # a = x[n, 0, :, :]
        # plt.imshow(a.cpu())
        # plt.show()
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def test_epoch_end(self, outputs):
        list_of_metrics = [output["metric"] for output in outputs]
        avg_metric = torch.stack(list_of_metrics).mean()
        self.log("metric", avg_metric)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def validation_epoch_end(self, outputs):
        print("Validation Finished\n")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
