import sys
sys.path.append('..')

from models import BoringModel
from datasets import RandomDataModule

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

# python -m main fit --config ./config.yaml

if __name__ == "__main__":
    cli = LightningCLI(
        BoringModel,
        RandomDataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None},
    )
