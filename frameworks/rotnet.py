import torch
import torch.nn as nn
import torch.nn.functional as F
from .supervised_learning import SupervisedLearning

class RotNet(SupervisedLearning):
    """RotNet framework - 논문 기반 구현"""
    def __init__(self, encoder):
        super().__init__(encoder)
        
        # Get feature dimension for rotation classifier
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_features = self.encoder._extract_features(dummy_input)
            feature_dim = dummy_features.size(1)
            
        # 4-class rotation classifier
        self.rotation_classifier = nn.Linear(feature_dim, 4)
        
    def create_rotation_batch(self, x):
        """논문 방식: 4개 회전 이미지를 동시에 배치에 포함"""
        batch_size = x.size(0)
        
        # Generate all 4 rotations
        x_rot0 = x  # 0°
        x_rot1 = torch.rot90(x, 1, dims=[2, 3])  # 90°
        x_rot2 = torch.rot90(x, 2, dims=[2, 3])  # 180°
        x_rot3 = torch.rot90(x, 3, dims=[2, 3])  # 270°
        
        # Stack all rotations (논문: feed simultaneously)
        x_all = torch.cat([x_rot0, x_rot1, x_rot2, x_rot3], dim=0)
        
        # Create rotation labels
        y_rot = torch.cat([
            torch.zeros(batch_size, dtype=torch.long),
            torch.ones(batch_size, dtype=torch.long),
            torch.full((batch_size,), 2, dtype=torch.long),
            torch.full((batch_size,), 3, dtype=torch.long)
        ], dim=0)
        
        return x_all, y_rot
        
    def forward(self, batch):
        """RotNet forward pass (논문 방식)"""
        if isinstance(batch, tuple):
            x, _ = batch  # Ignore original labels for SSL
        else:
            x = batch
            
        # Create rotation batch (논문: 4x more images than batch size)
        x_rot, y_rot = self.create_rotation_batch(x)
        x_rot = x_rot.to(x.device)
        y_rot = y_rot.to(x.device)
        
        # Extract features using encoder
        features = self.encoder._extract_features(x_rot)
        
        # Predict rotation
        rotation_pred = self.rotation_classifier(features)
        
        # Compute rotation loss
        loss = F.cross_entropy(rotation_pred, y_rot)
        
        return loss
        
    def extract_features(self, x):
        """Extract features for downstream tasks"""
        with torch.no_grad():
            features = self.encoder._extract_features(x)
            return features