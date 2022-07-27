import time
from pytorch_lightning.utilities.cli import LightningCLI
import sys
import pytorch_lightning
import torch
import traceback

sys.path.append("..")
from boring_model.boirng_main import RandomDataModule, RandomDataset, BoringModel
from utils.utils import *

# from loggers import SigOptLogger
# a = SigOptLogger('logs', name= 'e123', config_path='./pl-yaml/sigopt/egnn_sigopt.yml')

# python -m hack_main --config config.yaml


class CLI(LightningCLI):
    pass


def get_cli():
    return CLI(
        BoringModel,
        RandomDataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None},
        run=False,
    )


if __name__ == "__main__":
    try:
        cli = get_cli()
        cli.trainer.logger.experiment
        criterion = isinstance(
            cli.trainer.logger.experiment,
            pytorch_lightning.loggers.base.DummyExperiment,
        )
        cfgparser = ConfigParser()
        cfgparser.add_config("./config.yaml")
        if not criterion:
            soc = cli.trainer.logger._experiment.sigopt_context
            print(soc)
            print(cli.trainer.logger._experiment.experiment_dir)
            cfgparser.add_sigopt_hyperparams(soc.params)
            cli.og_config = cfgparser._config
            print(cfgparser._config)
            with open("current_sigopt_config.yaml", "w") as file:
                _ = yaml.dump(cfgparser._config, file, sort_keys=False)
        else:
            cfgparser = ConfigParser()
            cfgparser.add_config("current_sigopt_config.yaml")
        cfgparser.print_config()
        model = BoringModel(**cfgparser._config["model"])
        model.lr = cfgparser._config["optimizer"]["init_args"]["lr"]
        model.gamma = cfgparser._config["lr_scheduler"]["init_args"]["gamma"]
        cli.model = model
        datamodule = RandomDataModule(**cfgparser._config["data"])
        cli.datamodule = datamodule
        cli.trainer.fit(cli.model, cli.datamodule)
        if not criterion:
            soc.end()
    except Exception as e:
        try:
            print(traceback.format_exc())
            with open("error_log.txt", "a+") as file:
                file.write("\n" + e)
            soc.log_metadata("exception", e)
            soc.log_failure()
        except Exception:
            pass
