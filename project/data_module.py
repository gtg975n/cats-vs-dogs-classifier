import os
import torch
from pytorch_lightning import LightningDataModule
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image


class CatsDogsDataModule(LightningDataModule):
    def __init__(self, data_dir="data", batch_size=32, image_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

        # Define the transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images
            transforms.ToTensor(),  # Convert images to tensor
        ])

    def prepare_data(self):
        # No download needed — assumes images are already in `data/train` and `data/test1`
        pass

    def setup(self, stage=None):
        # Load the training dataset (cat and dog folders)
        train_dataset = ImageFolder(root=os.path.join(self.data_dir, "train"), transform=self.transform)

        # Split the train dataset into train and validation sets (80% train, 20% validation)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(train_dataset, [train_size, val_size])

        # Load the test dataset (unlabeled images in "test1")
        test_images = []
        test_dir = os.path.join(self.data_dir, "test1")
        for filename in os.listdir(test_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(test_dir, filename)
                image = Image.open(image_path)  # Open the image using PIL
                image = self.transform(image)  # Apply the transformation
                test_images.append(image)

        # Create a custom dataset for the test images (no labels)
        self.test_dataset = torch.utils.data.TensorDataset(torch.stack(test_images))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # validation data should NOT be shuffled
            num_workers=os.cpu_count(),  # improves speed on multi-core CPUs
            pin_memory=True  # helps if you're using a GPU
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
