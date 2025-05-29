from torchvision import datasets, transforms

def load_datasets(name, train=True) :
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    if name == "cifar10" :
        return datasets.CIFAR10(root = "./data", train=train, download=True, transform=transform)
    elif name == "cifar100" :
        return datasets.CIFAR100(root = "./data", train=train, download=True, transform=transform)