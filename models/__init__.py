from .resnet import resnet34
from .preactresnet import preactresnet
from .densenet import densenet
from .fractalnet import fractalnet
from .rotnet import rotnet

def load_model(model_name, num_classes=10, **kwargs):
    models = {
        'resnet34': resnet34,
        'preactresnet': preactresnet,
        'densenet': densenet,
        'fractalnet': fractalnet,
        'rotnet': rotnet
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](num_classes=num_classes, **kwargs)