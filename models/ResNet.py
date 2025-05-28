import torch
import torch.nn as nn

class ResidualBlock(nn.Module) :

    def __init__(self, in_channels, out_channels, stride=1) :
        super().__init__()
        self.relu = nn.ReLU()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x) :
        return self.relu(self.residual_block(x) + self.shortcut(x))

class BottleneckBlock(nn.Module) :

    def __init__(self, in_channels, out_channels, stride=1) :
        super().__init__()
        self.relu = nn.ReLU()

        self.bottleneck_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, 4*out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(4*out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != 4*out_channels :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, 4*out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(4*out_channels)
            )

    def forward(self, x) :
        return self.relu(self.bottleneck_block(x) + self.shortcut(x))

class ResNet34(nn.Sequential) :
    def __init__(self, num_classes=10) :
        super().__init__(
            #conv1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #conv2_X
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            #conv3_x
            ResidualBlock(64, 128, stride=2), 
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            #conv4_x
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            #conv5_x
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            #result
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

class ResNet50(nn.Sequential) :
    def __init__(self, num_classes=10) :
        super().__init__(
            #conv1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #conv2_x
            BottleneckBlock(64, 64),
            BottleneckBlock(256, 64),
            BottleneckBlock(256, 64),
            #conv3_x
            BottleneckBlock(256, 128, stride=2),
            BottleneckBlock(512, 128),
            BottleneckBlock(512, 128),
            BottleneckBlock(512, 128),
            #conv4_x
            BottleneckBlock(512, 256, stride=2),
            BottleneckBlock(1024, 256),
            BottleneckBlock(1024, 256),
            BottleneckBlock(1024, 256),
            BottleneckBlock(1024, 256),
            BottleneckBlock(1024, 256),
            #conv5_x
            BottleneckBlock(1024, 512, stride=2),
            BottleneckBlock(2048, 512),
            BottleneckBlock(2048, 512),
            #result
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

#model34 = ResNet34(num_classes=10)
#model50 = ResNet50(num_classes=10)