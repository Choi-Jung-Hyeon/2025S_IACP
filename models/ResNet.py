import torch
import torch.nn as nn

class ResidualBlock(nn.Module) :
    constant = 1
    def __init__(self, in_channels, out_channels, stride=1) :
        
        super().__init__()
        self.relu = nn.ReLU()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x) :
        return self.relu(self.residual_block(x) + self.shortcut(x))

class BottleneckBlock(nn.Module) :
    constant = 4
    def __init__(self, in_channels, out_channels, stride=1) :
        
        super().__init__()
        self.relu = nn.ReLU()

        self.bottleneck_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, self.constant*out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.constant*out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.constant*out_channels :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.constant*out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.constant*out_channels)
            )

    def forward(self, x) :
        return self.relu(self.bottleneck_block(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        self.conv2 = self._make_layer(block, 64, layers[0], stride=1)
        self.conv3 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv4 = self._make_layer(block, 256, layers[2], stride=2)
        self.conv5 = self._make_layer(block, 512, layers[3], stride=2)

        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(512 * block.constant, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.constant
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.feature_head(x)
        x = self.fc(x)
        return x

    def _extract_features(self, x):
        x = self.stem(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.feature_head(x)
        return x
    
def resnet18(num_classes=10):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=10):
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=10):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes)
