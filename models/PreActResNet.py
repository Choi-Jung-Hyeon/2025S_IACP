import torch
import torch.nn as nn

class PreActBlock(nn.Module) :

    def __init__(self, in_channels, out_channels, stride=1) :
        super().__init__()

        self.preAct_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            )

    def forward(self, x) :
        return (self.preAct_block(x) + self.shortcut(x))

class PreActResNet(nn.Sequential) :
    def __init__(self, num_classes=10) :
        super().__init__(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            PreActBlock(64, 64),
            PreActBlock(64, 64),
            PreActBlock(64, 128, stride=2),
            PreActBlock(128, 128),
            PreActBlock(128, 256, stride=2),
            PreActBlock(256, 256),
            PreActBlock(256, 512, stride=2),
            PreActBlock(512, 512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

#model = PreActResNet(num_classes=10)