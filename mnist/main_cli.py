from pytorch_lightning.utilities.cli import LightningCLI
from main import LitModel, MNISTDataModule

if __name__ == "__main__":
    cli = LightningCLI(
        LitModel, MNISTDataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None}
    )
    # cli = LightningCLI(
    #     LitModel, MNISTDataModule,
    #     save_config_callback=None,
    #     parser_kwargs={"error_handler": None},
    #     run=False
    # )
    # for _ in range(10):
    #     print(f"Training Loop: {_}")
    #     import pdb; pdb.set_trace()
    #     cli.trainer.fit(cli.model, cli.datamodule)
