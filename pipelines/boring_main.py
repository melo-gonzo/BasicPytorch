import sys

sys.path.append('..')

import yaml
from data_modules import RandomDataModule
from models import BoringModel
from pytorch_lightning import Trainer

# python -m main fit --config ./config.yaml

if __name__ == "__main__":

    with open("../configs/boring_model_config.yaml", 'r') as stream:
        config=yaml.safe_load(stream)

    model = BoringModel(**config['model'])
    datamodule=RandomDataModule(**config['data'])
    trainer = Trainer(**config['trainer'])
    trainer.fit(model, datamodule)


