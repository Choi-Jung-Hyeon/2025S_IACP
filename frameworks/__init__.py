import torch
import torch.nn as nn

class Framework(nn.Module):
    """Base framework class"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Identity()
        
    def forward(self, batch):
        """Framework specific forward pass"""
        raise NotImplementedError
        
    def extract_features(self, x):
        """Extract features using encoder"""
        return self.encoder(x)
        
    def remove_classifier(self):
        """Remove classifier for SSL feature extraction"""
        if hasattr(self.encoder, 'fc'):
            self.encoder.fc = nn.Identity()
        elif hasattr(self.encoder, 'classifier'):
            self.encoder.classifier = nn.Identity()

from .rotnet import RotNet
from .supervised_learning import SupervisedLearning