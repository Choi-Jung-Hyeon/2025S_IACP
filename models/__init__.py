from .ResNet import ResNet34, ResNet50
from .PreActResNet import PreActResNet

def load_model(model_name, num_classes=10) :
    if model_name == "resnet34" :
        return ResNet34(num_classes=num_classes)
    elif model_name == "resnet50" :
        return ResNet50(num_classes=num_classes)
    elif model_name == "preactresnet" :
        return PreActResNet(num_classes=num_classes)
    elif model_name == "densenet" :
        pass
    elif model_name == "fractalnet" :
        pass
