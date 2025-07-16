from .ResNet import ResNet18, ResNet34, ResNet50
from .DenseNet import DenseNet
from .FractalNet import FractalNet
from .PreActResNet import PreActResNet18
from .ViT import ViT
from .rotnet import rotnet


def load_model(model_name, data_name):
    data_name = data_name.lower()
    if data_name == "cifar100":
        num_classes = 100
    elif data_name == "cifar10":
        num_classes = 10
    else:
        num_classes = 10

    model_name = model_name.lower()
    if model_name == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif model_name == "resnet34":
        return ResNet34(num_classes=num_classes)
    elif model_name == "resnet50":
        return ResNet50(num_classes=num_classes)
    elif model_name == "preactresnet18":
        return PreActResNet18(num_classes=num_classes)
    elif model_name == "densenet":
        return DenseNet(num_classes=num_classes)
    elif model_name == "fractalnet":
        return FractalNet(num_classes=num_classes)
    elif model_name == "vit":
        return ViT(num_classes=num_classes)
    elif model_name == "rotnet":
        return rotnet(num_classes=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
