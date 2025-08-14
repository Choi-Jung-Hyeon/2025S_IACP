import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseFramework

class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.block(x)

class BYOL(BaseFramework):
    def __init__(self, encoder, total_steps, projection_size=256, projection_hidden_size=4096, base_moving_average_decay=0.996):
        super().__init__(encoder)
        
        self.base_moving_average_decay = base_moving_average_decay
        self.total_steps = total_steps
        
        with torch.no_grad():
            dummy = torch.randn(2, 3, 32, 32).to(next(encoder.parameters()).device)
            self.feature_dim = self.encoder._extract_features(dummy).shape[-1]

        self.online_projector = MLPHead(self.feature_dim, projection_hidden_size, projection_size)
        self.online_predictor = MLPHead(projection_size, projection_hidden_size, projection_size)

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        self._set_requires_grad(self.target_encoder, False)
        self._set_requires_grad(self.target_projector, False)

    def _set_requires_grad(self, model, val):
        for p in model.parameters():
            p.requires_grad = val

    @torch.no_grad()
    def _update_moving_average(self, step):
        # 코사인 스케줄링에 따른 EMA decay 계산
        tau = 1 - (1 - self.base_moving_average_decay) * (math.cos(math.pi * step / self.total_steps) + 1) / 2
        
        for online_p, target_p in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_p.data = target_p.data * tau + online_p.data * (1. - tau)
        
        for online_p, target_p in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_p.data = target_p.data * tau + online_p.data * (1. - tau)

    def _regression_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, batch, step):
        (view1, view2), _ = batch

        if self.training:
            self._update_moving_average(step)

        # Online Network
        online_feat1 = self.encoder._extract_features(view1)
        online_proj1 = self.online_projector(online_feat1)
        online_pred1 = self.online_predictor(online_proj1)

        online_feat2 = self.encoder._extract_features(view2)
        online_proj2 = self.online_projector(online_feat2)
        online_pred2 = self.online_predictor(online_proj2)

        # Target Network (stop-gradient)
        with torch.no_grad():
            target_proj1 = self.target_projector(self.target_encoder._extract_features(view1))
            target_proj2 = self.target_projector(self.target_encoder._extract_features(view2))
            
        # Symmetric Loss
        loss1 = self._regression_loss(online_pred1, target_proj2.detach())
        loss2 = self._regression_loss(online_pred2, target_proj1.detach())

        loss = loss1 + loss2
        return loss.mean()
    
    def extract_features(self, x):
        return self.encoder._extract_features(x)