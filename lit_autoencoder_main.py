# project/lit_autoencoder_main.py

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from project.data_module import CatsDogsDataModule
from project.lit_autoencoder import LitAutoEncoder
import json
import os

def main():
    config = {
        "model": "AutoEncoder+Classifier",
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 10,
        "latent_dim": 128,
        "classify": False #True,
    }

    os.makedirs("config", exist_ok=True)
    with open("config/autoencoder_supervised_config.json", "w") as f:
        json.dump(config, f, indent=4)

    data_module = CatsDogsDataModule(data_dir="data", batch_size=config["batch_size"])

    model = LitAutoEncoder(
        latent_dim=config["latent_dim"],
        classify=config["classify"],
        lr=config["learning_rate"]
    )

    monitor_metric = 'val_acc' if config["classify"] else 'val_loss'
    monitor_mode = 'max' if monitor_metric == 'val_acc' else 'min'

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath='checkpoints',
        filename=f'best_autoencoder_classifier-epoch={{epoch}}-{monitor_metric}={{{monitor_metric}:.2f}}',
        save_top_k=1,
        mode=monitor_mode
    )

    logger = CSVLogger("logs", name="autoencoder_supervised")

    trainer = Trainer(
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(model, datamodule=data_module)

    test_results = trainer.test(model, dataloaders=data_module.test_dataloader())
    with open("test_results_autoencoder_supervised.json", "w") as f:
        json.dump(test_results, f, indent=4)

        After
        training
    print("Testing whether encode() method exists in the loaded model...")

    loaded_model = LitAutoEncoder.load_from_checkpoint(checkpoint_callback.best_model_path)
    if hasattr(loaded_model, "encode"):
        print("✅ encode() method is available.")
    else:
        print("❌ encode() method is MISSING.")


if __name__ == "__main__":
    main()

#