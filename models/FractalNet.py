import torch
import torch.nn as nn
import random

def join(inputs):
    return torch.stack(inputs).mean(dim=0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))

class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, stride=1, drop_path=0.15):
        super().__init__()
        self.depth = depth
        self.drop_path = drop_path
        
        self.block0 = ConvBlock(in_channels, out_channels, stride)
        if depth > 1:
            self.block1 = FractalBlock(in_channels, out_channels, depth-1, stride, drop_path)
            self.block2 = FractalBlock(out_channels, out_channels, depth-1, 1, drop_path)
    
    def forward(self, x):
        y = [self.block0(x)]
        
        if self.depth > 1:
            z = join(self.block1(x))
            y.extend(self.block2(z))
        
        if self.training and self.drop_path > 0 and len(y) > 1:
            if random.random() < 0.5:
                # Local drop
                k = max(1, len(y) // 2)
                indices = random.sample(range(len(y)), k)
                y = [y[i] for i in sorted(indices)]
            else:
                # Global drop
                y = [y[random.randint(0, len(y) - 1)]]
        
        return y

class FractalNet(nn.Module):
    def __init__(self, num_classes=10, columns=4, blocks=5, drop_path=0.15):
        super().__init__()
        channels = [64, 128, 256, 512, 512]
        
        self.conv1 = ConvBlock(3, 64)
        
        self.blocks = nn.ModuleList()
        for i in range(blocks):
            in_ch = channels[i-1] if i > 0 else 64
            out_ch = channels[i]
            stride = 2 if i > 0 else 1
            self.blocks.append(FractalBlock(in_ch, out_ch, columns, stride, drop_path))
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        
        for block in self.blocks:
            x = join(block(x))
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def _extract_features(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = join(block(x))
        x = self.global_pool(x)
        return torch.flatten(x, 1)

def fractalnet(num_classes=10, **kwargs):
    return FractalNet(num_classes=num_classes)