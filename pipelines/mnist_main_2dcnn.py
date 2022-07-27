import sys

sys.path.append("..")
from pytorch_lightning import Trainer
import torch 

from models import Lit2DCNN
from data_modules import MNISTDataModule

import yaml
import time

if __name__ == "__main__":
    start_time = time.time()
    with open("../configs/mnist_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    model = Lit2DCNN(**config["model"])
    datamodule = MNISTDataModule(**config["data"])
    trainer = Trainer(**config["trainer"])
    trainer.fit(model, datamodule)

    end_time = torch.round(torch.tensor(time.time() - start_time)).item()
    print(f"5 Epoch Time: {end_time} seconds")

# python mnist_main_2dcnn.py

