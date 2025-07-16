import torch
import torch.nn as nn

class SupervisedLearning(nn.Module):
    """Base framework for supervised learning (7월 10일 Comment(1))"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Identity()  # Default
        
    def forward(self, batch):
        """Standard supervised forward pass"""
        if isinstance(batch, tuple):
            x, y = batch
        else:
            x = batch
            y = None
            
        output = self.encoder(x)
        
        if y is not None:
            # Training mode - compute loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, y)
            return loss
        else:
            # Inference mode
            return output
    
    def extract_features(self, x):
        """Extract features for downstream tasks"""
        with torch.no_grad():
            if hasattr(self.encoder, '_extract_features'):
                return self.encoder._extract_features(x)
            else:
                return self.encoder(x)
    
    def move_batch_to_device(self, batch, device):
        """Move batch to device (tuple 처리)"""
        if isinstance(batch, tuple):
            return tuple(item.to(device) if torch.is_tensor(item) else item for item in batch)
        else:
            return batch.to(device)