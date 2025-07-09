from torchvision import datasets, transforms

def load_dataset(name, train=True):
    # train/test에 따라 다른 transform 적용
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if name.lower() == "cifar10":
        return datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
    elif name.lower() == "cifar100":
        return datasets.CIFAR100(root="./data", train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset '{name}' is not supported.")
