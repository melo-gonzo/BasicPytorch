import os

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch import nn
from torch.utils.data import DataLoader, random_split
# from torchmetrics.functional import accuracy
import torchmetrics
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from pytorch_lightning.utilities.cli import LightningCLI


PATH_DATASETS = "./data"  # os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
# 256


class custom_augmentation(object):
    def __init__(self):
        pass

    def __call__(self, img):
        fft = torch.fft.fft2(img[0, :, :])
        angle = fft.angle()
        amp = fft.abs()
        img[0, :, :] = torch.mul(amp, angle)
        return img

    def __repr__(self):
        return "custom augmentation"


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir: str = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # custom_augmentation(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True,
                               transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)


class LitModel(LightningModule):
    def __init__(
        self, channels, width, height, num_classes, hidden_size=64, learning_rate=2e-4
    ):
        super().__init__()
        # We take in input dimensions as parameters and use those to dynamically build model.
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        accuracy = self.train_accuracy(logits, y)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def test_epoch_end(self, outputs):
        list_of_metrics = [output["metric"] for output in outputs]
        avg_metric = torch.stack(list_of_metrics).mean()
        self.log("metric", avg_metric)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def validation_epoch_end(self, outputs):
        print("Validation Finished\n")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.set_defaults({"trainer.max_epochs": 100})
        # pass


if __name__ == "__main__":
    cli = CLI(
        LitModel,
        MNISTDataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None},
    )

# python -m main fit --config ./test.yml
