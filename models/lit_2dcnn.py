
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
import torchmetrics


class Lit2DCNN(LightningModule):
    def __init__(
        self, channels, width, height, num_classes, hidden_size=64, learning_rate=2e-4
    ):
        super().__init__()
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
