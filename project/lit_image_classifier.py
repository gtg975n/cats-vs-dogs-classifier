from argparse import ArgumentParser
import os

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms, models

class LitClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Use pretrained ResNet18 for binary classification
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        # In the case of test data, there are no labels, so we only get the images (x)
        x = batch[0]  # Unpack only the images (no labels in test data)

        # Perform prediction
        logits = self(x)

        # We can't calculate loss or accuracy without labels in the test dataset
        return logits  # Just return the logits for the test data

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='data')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # transforms and dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(os.path.join(args.data_dir, "train"), transform=transform)

    # split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = LitClassifier(learning_rate=args.learning_rate)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # You could also add test set here if labeled, but "test1/" is usually unlabeled

if __name__ == '__main__':
    cli_main()
