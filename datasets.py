import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class SSLDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        return image

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

def load_dataset(dataset_name, train=True, ssl_mode=False):
    transform = get_transforms(train)
    
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10('./archive', train, transform=transform)
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100('./archive', train, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return SSLDataset(dataset) if ssl_mode and train else dataset

def get_num_classes(dataset_name):
    return 10 if dataset_name == "cifar10" else 100