import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class SSLDataset(Dataset):
    """Dataset wrapper for SSL mode (labels 제거)"""
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]  # Ignore labels for SSL
        return self.transform(image)

class SimCLRDataset(Dataset):
    """Dataset wrapper for SimCLR (두 개의 augmented views 생성)"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        size = 32
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),  # 쉼표 추가
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        x_i = self.transform(x)
        x_j = self.transform(x)
        return (x_i, x_j), 0

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

def load_dataset(dataset_name, train=True, ssl_framework=None):
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=None
        )
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root='./data', train=train, download=True, transform=None
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if ssl_framework in ['simclr', 'simsiam'] and train:
        dataset = SimCLRDataset(dataset)
    elif ssl_framework and train:  # 'rotnet' 등 다른 SSL 프레임워크
        transform = get_transforms(train=True)
        dataset = SSLDataset(dataset, transform=transform)
    else:  # 지도학습 또는 평가
        transform = get_transforms(train)
        dataset.transform = transform
    
    return dataset

def get_num_classes(dataset_name):
    return 10 if dataset_name == "cifar10" else 100
