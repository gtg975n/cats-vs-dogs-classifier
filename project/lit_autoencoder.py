# project/lit_autoencoder.py

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, latent_dim=128, classify=False, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.classify = classify
        self.lr = lr

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # [B, 32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # [B, 64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # [B, 128, 16, 16]
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [B, 64, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # [B, 32, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # [B, 3, 128, 128]
            nn.Sigmoid(),
        )

        if self.classify:
            # Classifier Head
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),        # [B, 128, 1, 1]
                nn.Flatten(),                        # [B, 128]
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)                     # Binary classification
            )
            self.train_acc = Accuracy(task="binary")
            self.val_acc = Accuracy(task="binary")

    def forward(self, x):
        z = self.encoder(x)
        if self.classify:
            return self.classifier(z)
        else:
            return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)

        if self.classify:
            logits = self.classifier(z)
            loss = F.cross_entropy(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = self.train_acc(preds, y)
            self.log("train_acc", acc, prog_bar=True)
            self.log("train_loss", loss, prog_bar=True)
        else:
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # validation batches have (x, y)
        x, y = batch
        z = self.encoder(x)

        if self.classify:
            logits = self.classifier(z)
            loss = F.cross_entropy(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = self.val_acc(preds, y)
            self.log("val_acc", acc, prog_bar=True)
            self.log("val_loss", loss, prog_bar=True)
        else:
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        # test batches have only x
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch  # just in case test has labels
        else:
            x = batch
            y = None

        z = self.encoder(x)

        if self.classify:
            logits = self.classifier(z)
            loss = F.cross_entropy(logits, y) if y is not None else torch.tensor(0.0, device=x.device)
            preds = torch.argmax(logits, dim=1)
            # you may want to log test accuracy only if y is available
            if y is not None:
                acc = self.val_acc(preds, y)
                self.log("test_acc", acc, prog_bar=True)
            self.log("test_loss", loss, prog_bar=True)
        else:
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            z = self.encoder(x)
            if z.ndim == 4:  # [B, C, H, W]
                z = F.adaptive_avg_pool2d(z, (1, 1))
                z = z.view(z.size(0), -1)  # [B, C]
            return z


