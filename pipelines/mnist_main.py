import sys

sys.path.append('..')

import yaml
from models import Basic2DCNN
from data_modules import MNISTDataModule
from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == "__main__":
    cli = LightningCLI(
        Basic2DCNN,
        MNISTDataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None},
    )

# python -m mnist_main fit --config ../configs/mnist_config.yaml
