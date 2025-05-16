import os
import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from project.data_module import CatsDogsDataModule
from project.lit_autoencoder import LitAutoEncoder

def extract_embeddings(model_path=None, split="train"):
    # Automatically detect the best checkpoint if not provided
    if model_path is None:
        checkpoint_dir = "checkpoints"
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]

        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}. Please train the autoencoder first.")

        # Extract epoch number and sort
        def extract_epoch_num(fname):
            match = re.search(r"epoch=(\d+)", fname)
            return int(match.group(1)) if match else -1

        ckpt_files.sort(key=extract_epoch_num)
        model_path = os.path.join(checkpoint_dir, ckpt_files[-1])  # Pick checkpoint with highest epoch

    print(f"Loading checkpoint: {model_path}")

    # Set device (MPS if available, otherwise CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize the data module
    dm = CatsDogsDataModule(data_dir="data", batch_size=32)
    dm.setup(stage=split)

    # Get the dataset based on the split (train, val, test)
    dataset = getattr(dm, f"{split}_dataset")
    loader = DataLoader(dataset, batch_size=32, drop_last=True)

    # Load the trained autoencoder and move it to the correct device
    model = LitAutoEncoder.load_from_checkpoint(model_path)
    model = model.to(device)
    model.eval()

    # Detect whether input should be flattened
    first_encoder_layer = list(model.encoder.children())[0]
    requires_flatten = isinstance(first_encoder_layer, torch.nn.Linear)
    #requires_flatten=0 #temporary
    print(f"Auto-detect flatten input: {'Yes' if requires_flatten else 'No'}")

    embeddings = []
    labels = [] if split != "test" else None

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {split} embeddings"):
            x = batch[0].to(device)
            y = batch[1] if split != "test" else None

            if requires_flatten:
                x = x.view(x.size(0), -1)

            z = model.encoder(x)

            if z is not None and z.ndim == 4:
                z = torch.nn.functional.adaptive_avg_pool2d(z, (1, 1))  # [B, C, 1, 1]
                z = z.view(z.size(0), -1)  # [B, C]

            if z is not None and z.shape[0] == x.shape[0]:  # Ensure batch size matches
                embeddings.append(z.cpu())
                if split != "test":
                    labels.append(y)

    embeddings = torch.cat(embeddings)

    if split != "test":
        labels = torch.cat(labels)
        assert embeddings.shape[0] == labels.shape[0], \
            f"Mismatch: embeddings={embeddings.shape[0]}, labels={labels.shape[0]}"
        return embeddings, labels
    else:
        return embeddings
