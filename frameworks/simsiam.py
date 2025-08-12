
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseFramework

class SimSiam(BaseFramework):
    def __init__(self, encoder, projection_dim=2048, prediction_dim=512):
        super().__init__(encoder)
        
        if hasattr(self.encoder, 'fc'):
            feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif hasattr(self.encoder, 'classifier'):
            feature_dim = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim, affine=False)
        )

        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, prediction_dim, bias=False),
            nn.BatchNorm1d(prediction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prediction_dim, projection_dim)
        )

    def forward(self, batch):
        (x1, x2), _ = batch

        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = -(F.cosine_similarity(p1, z2.detach()).mean() + F.cosine_similarity(p2, z1.detach()).mean()) * 0.5
        
        return loss
