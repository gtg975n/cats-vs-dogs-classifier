import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from project.data_module import CatsDogsDataModule  # Assuming you already have this data module

class MLPClassifier(pl.LightningModule):
    def __init__(self, input_dim=128):  # input_dim matches the latent size from autoencoder
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),  # First fully connected layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(64, 2)  # Output layer for binary classification (2 classes: cat vs. dog)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input (batch_size, input_dim) before passing to the MLP
        return self.fc(x)  # Forward pass through the MLP

    def training_step(self, batch, batch_idx):
        x, y = batch  # x is the embeddings, y is the labels
        logits = self(x)  # Forward pass through the model
        loss = F.cross_entropy(logits, y)  # Compute cross-entropy loss
        acc = (logits.argmax(dim=1) == y).float().mean()  # Compute accuracy
        self.log("train_loss", loss)  # Log the loss
        self.log("train_acc", acc)  # Log the accuracy
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)  # Use Adam optimizer with lr=1e-3

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

def main():
    # Hyperparameters (adjust as needed)
    config = {
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 10,
        "input_dim": 128,  # Should match the latent size from autoencoder
    }

    # Initialize the data module
    data_module = CatsDogsDataModule(data_dir="data", batch_size=config["batch_size"])

    # Initialize the model
    model = MLPClassifier(input_dim=config["input_dim"])

    # Define the ModelCheckpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Monitor validation loss or another metric
        dirpath='checkpoints',  # Save model weights in this folder
        filename='best_mlp_model_epoch={epoch}',  # Save model with epoch number
        save_top_k=1,
        mode='min',  # 'min' for loss, 'max' for accuracy
        save_weights_only=False,  # Save the full model
    )

    # Define the logger to track training metrics
    logger = CSVLogger("logs", name="mlp_comparison")

    # Initialize the trainer with callbacks and logger
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    # Fit the model
    trainer.fit(model, datamodule=data_module)

    # Test the model and save test results
    test_results = trainer.test(model, dataloaders=data_module.test_dataloader())

    # Save test results in a JSON file for later comparison
    with open("test_results_mlp.json", "w") as f:
        json.dump(test_results, f, indent=4)

if __name__ == "__main__":
    main()
