from .ResNet import resnet34
from .DenseNet import densenet
from .PreActResNet import preactresnet
from .FractalNet import fractalnet

def load_model(model_name, num_classes=10, **kwargs):
    """Load model with specified parameters"""
    model_name = model_name.lower()
    
    if model_name == "resnet34":
        return resnet34(num_classes=num_classes)
    elif model_name == "densenet":
        return densenet(num_classes=num_classes, **kwargs)
    elif model_name == "preactresnet":
        return preactresnet(num_classes=num_classes)
    elif model_name == "fractalnet":
        return fractalnet(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")