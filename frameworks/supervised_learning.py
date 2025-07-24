import torch.nn as nn
from .base import BaseFramework

class SupervisedLearning(BaseFramework):
    def forward(self, batch):
        x, y = batch if isinstance(batch, tuple) else (batch, None)
        output = self.encoder(x)
        
        if y is not None:
            return nn.CrossEntropyLoss()(output, y)
        return output