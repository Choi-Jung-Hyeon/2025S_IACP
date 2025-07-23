from .ResNet import resnet34, resnet18
from .DenseNet import densenet
from .PreActResNet import preactresnet
from .FractalNet import fractalnet
from .rotnet import rotnet
from .ViT import vit

def load_model(model_name, num_classes=10, **kwargs):
    models = {
        'resnet18' : resnet18,
        'resnet34': resnet34,
        'preactresnet': preactresnet,
        'densenet': densenet,
        'fractalnet': fractalnet,
        'vit' : vit,
        'rotnet': rotnet
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](num_classes=num_classes, **kwargs)
