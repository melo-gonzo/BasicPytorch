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



class ResNetTransfer(LightningModule):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        # build model
        model = models.resnet50(pretrained=True)
        num_filters = model.fc.in_features
        layers = list(model.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = num_classes
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        x = self.backbone(x)
        batch_size = x.size(0)
        x = F.relu(x.view(batch_size, -1))
        x = self.classifier(x)
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
