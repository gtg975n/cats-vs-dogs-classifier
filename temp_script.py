# project/lit_autoencoder_main.py

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from project.data_module import CatsDogsDataModule
from project.lit_autoencoder import LitAutoEncoder
import json
import os

model = LitAutoEncoder.load_from_checkpoint("checkpoints/best_model_epoch=epoch=6.ckpt")
data_module = CatsDogsDataModule(data_dir="data", batch_size=32)
trainer = Trainer()
trainer.test(model, dataloaders=data_module.test_dataloader())