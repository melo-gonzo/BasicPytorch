import sys
sys.path.append('..')
from pytorch_lightning.utilities.cli import LightningCLI

from models import Lit2DCNN
from data_modules import CIFAR10DataModule

if __name__ == "__main__":
    cli = LightningCLI(
        Lit2DCNN,
        CIFAR10DataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None},
    )

# python -m cifar_main_cli fit --config ../configs/cifar_test.yml
# python -m cifar_main_cli fit --config ../configs/cifar_test_ddp.yml