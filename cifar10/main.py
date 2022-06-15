import os
import torchvision
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch import nn
from torch.utils.data import DataLoader, random_split
# from torchmetrics.functional import accuracy
import torchmetrics
from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorch_lightning.utilities.cli import LightningCLI
import torchvision.models as models


PATH_DATASETS = "./data"  # os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
# 256


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir: str = data_dir
        self.num_classes = 10
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            CIFAR10_full = CIFAR10(self.data_dir, train=True,
                                   transform=self.train_transform)
            self.train_data, self.val_data = random_split(
                CIFAR10_full, [40000, 10000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = CIFAR10(
                self.data_dir, train=False, transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=BATCH_SIZE, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=BATCH_SIZE, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=BATCH_SIZE, num_workers=8)


class LitModel(LightningModule):
    def __init__(
        self, channels, width, height, num_classes, hidden_size=64, learning_rate=2e-4
    ):
        super().__init__()
        # We take in input dimensions as parameters and use those to dynamically build model.
        # self.channels = channels
        # self.width = width
        # self.height = height
        # self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        # build model
        self.c1 = nn.Conv2d(in_channels=3, out_channels=32,
                            kernel_size=3, padding=(1, 1), stride=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # 16x16
        self.c3 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.c4 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # 8x8

        self.c_d1 = nn.Linear(in_features=8*8*64,
                              out_features=256)
        self.c_d1_bn = nn.BatchNorm1d(256)
        self.c_d1_drop = nn.Dropout(0.5)

        self.c_d2 = nn.Linear(in_features=256,
                              out_features=10)

    def forward(self, x):

        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.pool1(x)  # 16
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = self.pool2(x)  # 8

        batch_size = x.size(0)
        x = F.relu(self.c_d1(x.view(batch_size, -1)))
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)

        x = self.c_d2(x)
        logits = F.log_softmax(x, dim=1)
        return logits

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
        CIFAR10DataModule,
        save_config_callback=None,
        parser_kwargs={"error_handler": None},
    )

# python -m main fit--config ./test.yml
