import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from project.lit_autoencoder import LitAutoEncoder
import torch.nn.functional as F  # <-- added import for pooling

class UnlabeledTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = read_image(img_path).float() / 255.0
        if self.transform:
            image = self.transform(image)
        image_id = int(os.path.splitext(self.image_filenames[idx])[0])
        return image, image_id

def predict_and_create_submission(

    model_type: str,
    ckpt_path: str = None,
    resnet_model=None,
    autoencoder=None,
    mlp_model=None,
    image_dir="data/test1",
    output_csv="submission.csv"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
    ])

    dataset = UnlabeledTestDataset(image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Sanity check dataloader
    sample_img, sample_id = next(iter(dataloader))
    print(f"Sample batch shape: {sample_img.shape}, sample ids: {sample_id}")

    if model_type == "autoencoder_classifier":
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        model = LitAutoEncoder.load_from_checkpoint(ckpt_path, classify=True)
        model.to(device).eval().freeze()

    elif model_type == "resnet":
        if resnet_model is None:
            raise ValueError("You must pass a ResNet model for model_type='resnet'")
        model = resnet_model.to(device).eval()

    elif model_type == "autoencoder_mlp":
        if autoencoder is None or mlp_model is None:
            raise ValueError("You must pass both autoencoder and mlp_model for model_type='autoencoder_mlp'")
        autoencoder = autoencoder.to(device).eval()
        mlp_model = mlp_model.to(device).eval()

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    all_preds = []
    all_ids = []

    with torch.no_grad():

        for x, ids in tqdm(dataloader, desc=f"Predicting with {model_type}"):
            x = x.to(device)
            #print(f"x shape before encoder: {x.shape}")  # should be [32, 3, 128, 128]

            if model_type == "autoencoder_classifier":
                logits = model(x)

            elif model_type == "resnet":
                logits = model(x)

            elif model_type == "autoencoder_mlp":
                z = autoencoder.encoder(x)
                #print(f"z shape after encoder: {z.shape}")  # e.g. [32, 128, 16, 16]
                # Add adaptive average pooling and flatten
                z = F.adaptive_avg_pool2d(z, (1, 1))  # [32, 128, 1, 1]
                z = torch.flatten(z, 1)  # [32, 128]
                #print(f"z shape after adaptive pooling and flatten: {z.shape}")
                logits = mlp_model(z)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_ids.extend(ids.numpy())

    submission = pd.DataFrame({
        "id": all_ids,
        "label": all_preds
    }).sort_values("id")
    submission.to_csv(output_csv, index=False)
    print(f"[✓] Submission file saved to {output_csv}")
