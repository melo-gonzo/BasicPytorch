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

PATH_DATASETS = "./data"  # os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
SCALE = 3.5
# 256

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.set_defaults({"trainer.max_epochs": 100})
        # pass


if __name__ == "__main__":
    start_time = time.time()
    with open("./config.yaml", 'r') as stream:
        config=yaml.safe_load(stream)

    model = LitModel(**config['model'])
    datamodule=MNISTDataModule(**config['data'])
    trainer = Trainer(**config['trainer'])
    trainer.fit(model, datamodule)

#    cli = CLI(
#        LitModel,
#        MNISTDataModule,
#        save_config_callback=None,
#        parser_kwargs={"error_handler": None},
#    )
    end_time = torch.round(torch.tensor(time.time()-start_time)).item()
    print(F"5 Epoch Time: {end_time} seconds")

# python -m main fit --config ./config.yml
