import sys

sys.path.append("..")

from data_modules import RandomDataModule
from models import BoringModel
from pytorch_lightning.utilities.cli import LightningCLI


if __name__ == "__main__":
    cli = LightningCLI(
        BoringModel,
        RandomDataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None},
    )

# python -m boring_main_cli fit --config ../configs/boring_model_config.yaml
# python -m boring_main_cli fit --config ../configs/boring_model_ddp_config.yaml
