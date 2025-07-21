import torch
import torch.nn as nn

class NINBlock(nn.Module):
    """Network-in-Network block: 3×3 conv + 2×1×1 conv (논문 기준)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class RotNet(nn.Module):
    """RotNet with NIN architecture (논문 기준)"""
    def __init__(self, num_classes=4, num_blocks=4):
        super().__init__()
        
        # 논문 기준 채널 구성
        if num_blocks == 3:
            channels = [96, 192, 192]
        elif num_blocks == 4:
            channels = [96, 192, 192, 192]
        elif num_blocks == 5:
            channels = [96, 192, 192, 192, 192]
        else:
            raise ValueError(f"Unsupported num_blocks: {num_blocks}")
        
        self.blocks = nn.ModuleList()
        
        # Block 1: 3 -> 96 (96 × 16 × 16)
        self.blocks.append(NINBlock(3, channels[0]))
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        # Block 2: 96 -> 192 (192 × 8 × 8)
        self.blocks.append(NINBlock(channels[0], channels[1]))
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        # Additional blocks (모두 192 채널)
        for i in range(2, num_blocks):
            self.blocks.append(NINBlock(channels[i-1], channels[i]))
        
        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)
        
    def forward(self, x):
        features = self._extract_features(x)
        return self.classifier(features)
    
    def _extract_features(self, x):
        # Block 1: 32×32 -> 16×16
        x = self.blocks[0](x)
        x = self.pool1(x)
        
        # Block 2: 16×16 -> 8×8
        x = self.blocks[1](x)
        x = self.pool2(x)
        
        # Additional blocks (8×8 유지)
        for i in range(2, len(self.blocks)):
            x = self.blocks[i](x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

def rotnet(num_classes=4, num_blocks=4):
    """Create RotNet model (논문 기준 NIN)"""
    return RotNet(num_classes=num_classes, num_blocks=num_blocks)