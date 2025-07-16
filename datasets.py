import torch
import torchvision
import torchvision.transforms as transforms

def get_transforms(dataset_name, train=True, ssl_mode=False):
    """Get transforms for dataset"""
    if dataset_name.lower() in ["cifar10", "cifar100"]:
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2470, 0.2435, 0.2616))  
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2470, 0.2435, 0.2616))  
            ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    return transform

def load_dataset(dataset_name, train=True, ssl_mode=False):
    """Load dataset"""
    transform = get_transforms(dataset_name, train, ssl_mode)
    
    if dataset_name.lower() == "cifar10":
        return torchvision.datasets.CIFAR10(
            root='./archive', train=train, download=True, transform=transform
        )
    elif dataset_name.lower() == "cifar100":
        return torchvision.datasets.CIFAR100(
            root='./archive', train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")