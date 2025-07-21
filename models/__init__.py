from .ResNet import resnet34, resnet18
from .DenseNet import densenet
from .PreActResNet import preactresnet
from .FractalNet import fractalnet
from .rotnet import rotnet
from .ViT import ViT

def load_model(model_name, num_classes=10, **kwargs):
    """Load model with specified parameters"""
    model_name = model_name.lower()
    
    if model_name == "resnet34":
        return resnet34(num_classes=num_classes)
    elif model_name == "resnet18":
        return resnet18(num_classes=num_classes)
    elif model_name == "densenet":
        return densenet(num_classes=num_classes, **kwargs)
    elif model_name == "preactresnet":
        return preactresnet(num_classes=num_classes)
    elif model_name == "fractalnet":
        return fractalnet(num_classes=num_classes, **kwargs)
    elif model_name == "rotnet":
        num_blocks = kwargs.get('num_blocks', 4)
        return rotnet(num_classes=num_classes, num_blocks=num_blocks)
    elif model_name == "vit":
        return ViT(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
