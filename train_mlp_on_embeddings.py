from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import TensorDataset, DataLoader
from extract_embeddings import extract_embeddings
from project.lit_mlp_classifier import MLPClassifier
import torch
import os

def main():
    # Extract embeddings for the training and validation sets
    X_train, y_train = extract_embeddings(split="train")
    print("X_train shape:", X_train.shape)
    X_val, y_val = extract_embeddings(split="val")

    # Ensure that the embeddings and labels are tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoader for training and validation
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    # Initialize MLP classifier model
    model = MLPClassifier(input_dim=X_train.shape[1])

    # Create directory for checkpoints if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='best_mlp_model_epoch={epoch}',
        save_top_k=1,
        mode='min'
    )

    # Setup logger (optional but useful for tracking)
    logger = CSVLogger("logs", name="mlp_classifier")

    # Initialize Trainer with callbacks and logger
    trainer = Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        logger=logger
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
