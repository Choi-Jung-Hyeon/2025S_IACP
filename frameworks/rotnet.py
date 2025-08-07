import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseFramework

class Rotnet(BaseFramework):
    def __init__(self, encoder):
        super().__init__(encoder)
        self._remove_classifier()
    
    def _remove_classifier(self):
        # Check if encoder has classifier/fc and extract feature dimension
        if hasattr(self.encoder, 'classifier') and not isinstance(self.encoder.classifier, nn.Identity):
            self.feature_dim = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()
        elif hasattr(self.encoder, 'fc') and not isinstance(self.encoder.fc, nn.Identity):
            self.feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            # For models without classifier or already Identity
            with torch.no_grad():
                dummy = torch.randn(1, 3, 32, 32)
                if hasattr(self.encoder, '_extract_features'):
                    self.feature_dim = self.encoder._extract_features(dummy).size(1)
                else:
                    # Forward pass to get output dimension
                    output = self.encoder(dummy)
                    if isinstance(output, tuple):
                        output = output[0]
                    self.feature_dim = output.size(1) if len(output.shape) == 2 else output.view(output.size(0), -1).size(1)
        
        # 4-way rotation classifier
        self.rotation_classifier = nn.Linear(self.feature_dim, 4)
    
    def forward(self, batch):
        # Handle different batch formats
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2 and torch.is_tensor(batch[0]):
                x = batch[0]  # (data, label) format
            elif torch.is_tensor(batch):
                x = batch
            else:
                # If batch is a list of tensors, stack them
                x = torch.stack(batch) if all(torch.is_tensor(b) for b in batch) else batch[0]
        else:
            x = batch
        
        # Ensure x is a tensor
        if not torch.is_tensor(x):
            if isinstance(x, list):
                x = torch.stack(x)
            else:
                raise TypeError(f"Expected tensor or list of tensors, got {type(x)}")
        
        # Create 4 rotations (0, 90, 180, 270 degrees)
        batch_size = x.size(0)
        x_rot = torch.cat([
            x,
            torch.rot90(x, 1, dims=[2, 3]),
            torch.rot90(x, 2, dims=[2, 3]),
            torch.rot90(x, 3, dims=[2, 3])
        ], dim=0)
        
        # Labels for rotations
        y_rot = torch.cat([
            torch.zeros(batch_size, dtype=torch.long),
            torch.ones(batch_size, dtype=torch.long),
            torch.full((batch_size,), 2, dtype=torch.long),
            torch.full((batch_size,), 3, dtype=torch.long)
        ]).to(x.device)
        
        # Extract features and predict rotation
        features = self.extract_features(x_rot)
        pred = self.rotation_classifier(features)
        
        return F.cross_entropy(pred, y_rot)
    
    def extract_features(self, x):
        if hasattr(self.encoder, '_extract_features'):
            return self.encoder._extract_features(x)
        else:
            # For models without _extract_features
            output = self.encoder(x)
            if isinstance(output, tuple):
                return output[0]
            return output.view(output.size(0), -1) if len(output.shape) > 2 else output