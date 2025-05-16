# project/lit_classifier_main.py

from project.lit_image_classifier import LitClassifier
from project.data_module import CatsDogsDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import json
import os

def main():
    # Hyperparameters (add any that you want to track)
    config = {
        "model": "ResNet18",  # You can replace this with your actual model
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 10,
    }

    # Save config to a JSON file (ensure the directory exists before saving)
    if not os.path.exists("config"):
        os.makedirs("config")
    with open("config/model_config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Initialize the data module
    data_module = CatsDogsDataModule(data_dir="data", batch_size=config["batch_size"])

    # Initialize the model
    model = LitClassifier()

    # Define the ModelCheckpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Can change to 'val_accuracy' or another metric if preferred
        dirpath='checkpoints',  # Save model weights in this folder
        filename='best_model_epoch={epoch}',  # Save model with epoch number
        save_top_k=1,
        mode='min',  # 'min' for loss, 'max' for accuracy
        save_weights_only=True,  # Save only weights, not the full model
    )

    # Define the logger to track training metrics
    logger = CSVLogger("logs", name="model_comparison")

    # Initialize the trainer with callbacks and logger
    trainer = Trainer(
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    # Fit the model
    trainer.fit(model, datamodule=data_module)

    # Test the model and save test results
    test_results = trainer.test(model, dataloaders=data_module.test_dataloader())

    # Save test results in a JSON file for later comparison
    with open("test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)

if __name__ == "__main__":
    main()
