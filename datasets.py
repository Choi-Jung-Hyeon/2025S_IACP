import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class SSLDataset(Dataset):
    """Dataset wrapper for SSL mode (labels 제거)"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]  # Ignore labels for SSL
        return image

def load_dataset(dataset_name, train=True, ssl_mode=False):
    """Load dataset with proper augmentation (6월 5일 comment)"""
    dataset_name = dataset_name.lower()
    
    if train:
        # Training augmentation
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        # Test augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root='./archive', train=train, download=False, transform=transform
        )
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root='./archive', train=train, download=False, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # SSL mode: remove labels
    if ssl_mode and train:
        dataset = SSLDataset(dataset)
    
    return dataset

def get_num_classes(dataset_name):
    """Get number of classes for dataset"""
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        return 10
    elif dataset_name == "cifar100":
        return 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")