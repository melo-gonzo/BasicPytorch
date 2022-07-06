import os
import torchvision
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch import nn
from torch.utils.data import DataLoader, random_split
# from torchmetrics.functional import accuracy
import torchmetrics
from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorch_lightning.utilities.cli import LightningCLI
import torchvision.models as models




class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.set_defaults({"trainer.max_epochs": 100})
        # pass


if __name__ == "__main__":
    cli = CLI(
        LitModel,
        CIFAR10DataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None},
    )

# python -m main fit --config ./test.yml
