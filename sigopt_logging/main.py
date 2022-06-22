import time
from pytorch_lightning.utilities.cli import LightningCLI
import pretty_errors
import sys
sys.path.append('..')
from boring_model.main import RandomDataModule, RandomDataset, BoringModel
from utils.utils import *

# python -m main --config config.yaml

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.set_defaults({"trainer.max_epochs": 10})


if __name__ == "__main__":
    cli = CLI(
        BoringModel,
        RandomDataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None},
        run=False,
        # logger=False
    )

    cli.trainer.logger.experiment
    if cli.trainer.logger.experiment is not None:
        cli.trainer.logger._experiment.create_experiment()
        experiment = cli.trainer.logger._experiment

        while not experiment.sigopt_experiment.is_finished():
            with experiment.sigopt_experiment.create_run() as soc:
                print('here')
                cli.trainer.logger._experiment.sigopt_context = soc
                print(cli.trainer.logger._experiment.experiment_dir)
                cfgparser = ConfigParser()
                cfgparser.add_config("./config.yaml")
                if soc is not None:
                    cfgparser.add_sigopt_hyperparams(soc.params)
                cfgparser.print_config()
                model = BoringModel(**cfgparser._config["model"])
                model.lr = cfgparser._config["optimizer"]["init_args"]["lr"]
                model.gamma = cfgparser._config["lr_scheduler"]["init_args"]["gamma"]
                cli.model = model
                datamodule = RandomDataModule(**cfgparser._config["data"])
                cli.datamodule = datamodule
                experiment_start_time = time.time()
                cli.trainer.fit(cli.model, cli.datamodule)
                test_time = time.time()
                soc.log_metrics({"test_time": time.time() - test_time})
                experiment_end_time = time.time() - experiment_start_time
                soc.log_metrics({"experiment_time": experiment_end_time})

            del cli
            cli = CLI(
                BoringModel,
                RandomDataModule,
                save_config_callback=None,
                parser_kwargs={"error_handler": None},
                run=False,
                # logger=False
            )
            cli.trainer.logger.experiment
            cli.trainer.logger._experiment = experiment

    # trainer_args = dict(max_epochs=4, logger=False, checkpoint_callback=False)
    # trainer = pl.Trainer(**trainer_args)
    # test_module = BoringModel()
    # trainer.fit(test_module, datamodule=RandomDataModule())