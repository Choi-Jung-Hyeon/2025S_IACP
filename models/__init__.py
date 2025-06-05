from .resnet34 import ResNet34
from .densenet import DenseNet
from .fractalnet import FractalNet
from .preactresnet import PreActResNet


def load_model(model_name, num_classes=10):
    model_name = model_name.lower()
    if model_name == "resnet34":
        return ResNet34(num_classes=num_classes)
    elif model_name == "preactresnet":
        return PreActResNet(num_classes=num_classes)
    elif model_name == "densenet" or model_name == "desnetnet":
        return DenseNet(num_classes=num_classes)
    elif model_name == "fractalnet":
        return FractalNet(num_classes=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
