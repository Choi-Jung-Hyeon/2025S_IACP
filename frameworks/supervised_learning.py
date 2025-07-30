import torch.nn as nn
from .base import BaseFramework

class SupervisedLearning(BaseFramework):
    def __init__(self, encoder):
        super().__init__(encoder)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        x, y = batch if isinstance(batch, tuple) else (batch, None)
        output = self.encoder(x)
        
        if y is not None:
            return self.criterion(output, y)
        return output
