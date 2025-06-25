from .resnet import resnet18, resnet34, resnet50
from .preactresnet import preactresnet18
from .densenet import DenseNet
from .fractalnet import FractalNet

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
        return resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        return resnet34(num_classes=num_classes)
    elif model_name == "resnet50":
        return resnet50(num_classes=num_classes)
    elif model_name == "preactresnet18":
        return preactresnet18(num_classes=num_classes)
    elif model_name == "densenet":
        return DenseNet(num_classes=num_classes)
    elif model_name == "fractalnet":
        return FractalNet(num_classes=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
