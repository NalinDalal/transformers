"""
ViT Dataset Utilities
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Optional, Tuple


def get_cifar10_dataset(
    train: bool = True,
    data_dir: str = "./data",
    download: bool = True
) -> datasets.CIFAR10:
    """Get CIFAR-10 dataset."""
    transform = get_cifar10_transforms(train)
    return datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )


def get_cifar10_transforms(train: bool = True) -> transforms.Compose:
    """Get transforms for CIFAR-10 resized to 224x224 for ViT."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_mnist_dataset(
    train: bool = True,
    data_dir: str = "./data",
    download: bool = True
) -> datasets.MNIST:
    """Get MNIST dataset (converted to 3 channels for ViT)."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return datasets.MNIST(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )


class DummyDataset(Dataset):
    """Dummy dataset for quick testing without downloading data."""
    
    def __init__(
        self, 
        num_samples: int = 100, 
        num_classes: int = 10,
        image_size: Tuple[int, int, int] = (3, 224, 224)
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random image and label
        image = torch.randn(self.image_size)
        label = idx % self.num_classes
        return image, label


def create_dummy_dataloader(
    num_samples: int = 100,
    num_classes: int = 10,
    batch_size: int = 8,
    image_size: Tuple[int, int, int] = (3, 224, 224)
):
    """Create a dummy dataloader for quick testing."""
    dataset = DummyDataset(num_samples, num_classes, image_size)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )


if __name__ == "__main__":
    # Test with dummy data
    from vit_model import ViTConfig, count_parameters
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ViTConfig.ViT_B_16(num_classes=10).to(device)
    print(f"ViT-B/16 parameters: {count_parameters(model):,}")
    
    # Test with dummy dataloader
    dataloader = create_dummy_dataloader(num_samples=20, batch_size=4)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
        break
    
    print("Dataset utilities test passed!")