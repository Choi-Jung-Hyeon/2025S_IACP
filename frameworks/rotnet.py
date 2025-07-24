import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseFramework

class Rotnet(BaseFramework):
    def _remove_classifier(self):
        if hasattr(self.encoder, 'classifier'):
            self.feature_dim = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()
        elif hasattr(self.encoder, 'fc'):
            self.feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            with torch.no_grad():
                dummy = torch.randn(1, 3, 32, 32)
                self.feature_dim = self.encoder._extract_features(dummy).size(1)
        
        self.rotation_classifier = nn.Linear(self.feature_dim, 4)
    
    def forward(self, batch):
        x = batch[0] if isinstance(batch, tuple) else batch
        
        # Create rotations
        batch_size = x.size(0)
        x_rot = torch.cat([
            x,
            torch.rot90(x, 1, dims=[2, 3]),
            torch.rot90(x, 2, dims=[2, 3]),
            torch.rot90(x, 3, dims=[2, 3])
        ], dim=0)
        
        y_rot = torch.cat([
            torch.zeros(batch_size, dtype=torch.long),
            torch.ones(batch_size, dtype=torch.long),
            torch.full((batch_size,), 2, dtype=torch.long),
            torch.full((batch_size,), 3, dtype=torch.long)
        ]).to(x.device)
        
        features = self.encoder._extract_features(x_rot)
        pred = self.rotation_classifier(features)
        return F.cross_entropy(pred, y_rot)