import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Fractal block with proper branching
class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, stride=1):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(0.5)
        
        self.block = ConvBlock(in_channels, out_channels, stride)
        if depth != 1:
            self.branch1 = FractalBlock(in_channels, out_channels, depth-1, stride)
            self.branch2 = FractalBlock(in_channels, out_channels, depth-1, stride)
    
    def forward(self, x):
        if self.depth == 1:
            return self.block(x)
        
        out1 = self.block(x)
        out1 = self.dropout(out1)
        out2 = self.branch2(self.branch1(x))
        out2 = self.dropout(out2)
        return (out1 + out2) / self.depth

# Main FractalNet
class FractalNet(nn.Module):
    def __init__(self, num_classes=10, num_columns=5, depth = 4):
        super().__init__()
        self.num_columns = num_columns
        self.depth = depth
        
        # Initial conv
        self.conv1 = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        # Fractal blocks (여러 개 사용)
        self.blocks = nn.ModuleList()
        in_channels = 64
        for _ in range(self.num_columns):
            self.blocks.append(FractalBlock(64, 64, depth, stride=1))
            self.blocks.append(self.pool)
            
    def forward(self, x):
        x = self.conv1(x)
        
        # Forward through fractal blocks
        for block in self.blocks:
            x = block(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# test
if __name__ == "__main__":
    model = FractalNet(num_classes=100, num_columns=5, depth=4)
    print(model)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
