import sys

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import torch 

# TODO move this to utils and get working
PATH_DATASETS = "../data"  

class custom_augmentation(object):
    def __init__(self):
        pass

    def __call__(self, img):
        fft = torch.fft.fft2(img[0, :, :])
        angle = fft.angle()
        amp = fft.abs()
        img[0, :, :] = torch.mul(amp, angle)
        # img = transforms.functional.equalize(img.byte()).float()
        return img

    def __repr__(self):
        return "custom augmentation"


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str):
        super().__init__()
        self.scale = 3.5
        self.batch_size = 256
        self.path_datasets = "../data/"
        self.data_dir: str = data_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((int(280 / self.scale), int(280 / self.scale))),
                transforms.ToTensor(),
                custom_augmentation(),
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
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
