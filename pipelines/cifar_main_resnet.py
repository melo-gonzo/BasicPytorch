import sys
sys.path.append('..')

from models import ResNetTransfer
from data_modules import CIFAR10DataModule

from pytorch_lightning.utilities.cli import LightningCLI


if __name__ == "__main__":
    cli = LightningCLI(
        ResNetTransfer,
        CIFAR10DataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None},
    )

# python -m cifar_main_resnet fit --config ../configs/cifar_test.yml
# python -m cifar_main_resnet fit --config ../configs/cifar_test_ddp.yml