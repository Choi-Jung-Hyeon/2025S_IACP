from torchvision import datasets, transforms

def load_dataset(name, train=True):
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
