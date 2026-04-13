import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms


def load_fashion_mnist(data_path="./data", batch_size=64, val_split=0.2):
    """Download and prepare Fashion-MNIST with train/validation split.

    Args:
        data_path: directory to store the downloaded dataset
        batch_size: number of samples per batch
        val_split: fraction of training data to use for validation

    Returns:
        train_loader, val_loader: DataLoader objects for training and validation sets
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    print("Loading Fashion-MNIST dataset...")
    full_dataset = datasets.FashionMNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform,
    )

    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_set, val_set = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"  Train: {n_train} samples | Val: {n_val} samples")
    return train_loader, val_loader
