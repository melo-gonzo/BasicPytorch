import operator
import time
import traceback

# from argparse import ArgumentParsern
from functools import reduce
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningCLI
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, size, length):
        super().__init__()
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class RandomDataModule(pl.LightningDataModule):
    def __init__(self, size, length, batch_size):
        super().__init__()
        self.size = size
        self.length = length
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        """
        Args:
            stage: used to separate setup logic for trainer.{fit,validate,test}.
                If setup is called with stage = None, we assume all stages have been set-up.
        """
        # if stage in (None, "fit"):
        self.train_dataset = RandomDataset(self.size, self.length)

        # if stage in (None, "test"):
        self.test_dataset = RandomDataset(self.size, self.length)

        # if stage in (None, "val"):
        self.val_dataset = RandomDataset(self.size, self.length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class BoringModel(LightningModule):
    def __init__(self, channels: int):
        super().__init__()
        self.layer = torch.nn.Linear(channels, 2)

    def forward(self, x):
        return self.layer(x)

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
        self.log("valid_loss", loss, prog_bar=True)

    def validation_epoch_end(self, outputs):
        print("validation finished")

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def get_from_dict(input_dict, map_list):
    return reduce(operator.getitem, map_list, input_dict)


def set_in_dict(input_dict, map_list, value):
    get_from_dict(input_dict, map_list[:-1])[map_list[-1]] = value


class ConfigParser:
    def __init__(self) -> None:
        self._config = None

    @property
    def config(self) -> Dict[str, Dict[str, Union[str, int, float]]]:
        return self._config

    def print_config(self):
        print("\n### CONFIG ###\n")
        print(self._config)
        print("\n#############\n")

    def add_config(self, path: str) -> None:
        with open(path) as f:
            self._config = yaml.safe_load(f)

    def add_sigopt_hyperparams(self, params: Dict[str, Union[str, int, float]]) -> None:
        assert (
            self._config is not None
        ), "Config should be provided before adding SigOpt hyperparameters."

        for k, v in params.items():
            keys = k.split(".")
            if keys[-1] == "depth":
                depth = v
                keys.pop(-1)
            else:
                depth = None

            if (
                isinstance(get_from_dict(self._config, keys), list)
                and depth is not None
            ):
                values = [params[".".join(keys)] for _ in range(depth)]
                set_in_dict(self._config, keys, values)

            elif not isinstance(get_from_dict(self._config, keys), list):
                set_in_dict(self._config, keys, v)


def run_with_sigopt(cli, sigopt_logger):
    cfgparser = ConfigParser()
    cfgparser.add_config("./config.yaml")

    if sigopt_logger.sigopt_context is not None:
        cfgparser.add_sigopt_hyperparams(sigopt_logger.sigopt_context.params)

    cfgparser.print_config()
    model = BoringModel()
    model.lr = cfgparser._config["optimizer"]["init_args"]["lr"]
    model.gamma = cfgparser._config["lr_scheduler"]["init_args"]["gamma"]
    cli.model = model
    experiment_start_time = time.time()
    # import pdb; pdb.set_trace()
    cli.trainer.fit(cli.model, cli.datamodule)
    test_time = time.time()
    sigopt_logger.log_metrics({"test_time": time.time() - test_time})
    experiment_end_time = time.time() - experiment_start_time
    sigopt_logger.log_metrics({"experiment_time": experiment_end_time})


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.set_defaults({"trainer.max_epochs": 10})


if __name__ == "__main__":
    cli = LightningCLI(
        BoringModel,
        RandomDataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None},
        run=False,
    )
    for _ in range(10):
        del cli
        cli = LightningCLI(
            BoringModel,
            RandomDataModule,
            save_config_callback=None,
            parser_kwargs={"error_handler": None},
            run=False,
        )
        print(f"Training Loop: {_}")
        cli.trainer.fit(cli.model, cli.datamodule)


# python other_main.py --config=config.yaml
# model:
#   channels: 32
# data:
#   size: 32
#   length: 256
#   batch_size: 2
# trainer:
#  max_epochs: 3
#  enable_checkpointing: False
#  accelerator: "gpu"
#  gpus: 1
# seed_everything: 42
