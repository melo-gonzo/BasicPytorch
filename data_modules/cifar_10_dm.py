
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = 10
        scale = 1
        # TOOD: Fix this image_dim hack to work with model and dm in sync
        image_dim = 32
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (int(256/scale), int(256/scale))),
                torchvision.transforms.RandomCrop(
                    (int(image_dim/scale), int(image_dim/scale))),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.244, 0.225]),
            ]
        )
        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (int(256/scale), int(256/scale))),
                torchvision.transforms.CenterCrop(
                    (int(image_dim/scale), int(image_dim/scale))),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.244, 0.225]),
            ]
        )

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
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=8)
