from .resnet34 import resnet34
from .densenet import DenseNet
from .fractalnet import FractalNet
from .preactresnet import PreActResNet
from .rotnet import rotnet

def load_model(model_name, num_classes=10):
    model_name = model_name.lower()
    if model_name == "resnet34":
        return resnet34(num_classes=num_classes)
    elif model_name == "preactresnet":
        return PreActResNet(num_classes=num_classes)
    elif model_name == "densenet":
        return DenseNet(num_classes=num_classes)
    elif model_name == "fractalnet":
        return FractalNet(num_classes=num_classes)
    elif model_name == "rotnet":
        return rotnet(num_classes=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")