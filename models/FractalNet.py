import torch
import torch.nn as nn
import torch.nn.functional as F

def join(inputs):
    """Join multiple tensors by averaging"""
    return torch.stack(inputs).mean(dim=0)

class ConvBlock(nn.Module):
    """Basic convolution block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FractalBlock(nn.Module):
    """Fractal block returning depth-length tensor list"""
    def __init__(self, in_channels, out_channels, depth, stride=1, drop_path=0.15):
        super().__init__()
        self.depth = depth
        self.drop_path = drop_path
        
        # Base block (always present)
        self.block0 = ConvBlock(in_channels, out_channels, stride)
        
        if depth > 1:
            # Two sub-blocks for fractal structure
            self.block1 = FractalBlock(in_channels, out_channels, depth-1, stride, drop_path)
            self.block2 = FractalBlock(out_channels, out_channels, depth-1, 1, drop_path)
            
    def forward(self, x):
        # Base path
        y = [self.block0(x)]
        
        if self.depth > 1:
            # Left branch
            branch1_outputs = self.block1(x)
            z = join(branch1_outputs)
            
            # Right branch  
            branch2_outputs = self.block2(z)
            
            # Extend with branch outputs
            y.extend(branch2_outputs)
            
        return y  # Returns depth-length tensor list

class FractalNet(nn.Module):
    """FractalNet implementation"""
    def __init__(self, num_classes=10, columns=4, blocks=5):
        super().__init__()
        self.columns = columns
        self.blocks = blocks
        
        # Initial conv
        self.conv1 = ConvBlock(3, 64)
        
        # Fractal blocks
        self.fractal_blocks = nn.ModuleList()
        in_channels = 64
        out_channels = 64
        
        for i in range(blocks):
            # Increase channels at certain blocks
            if i in [1, 3]:
                out_channels *= 2
                stride = 2
            else:
                stride = 1
                
            block = FractalBlock(in_channels, out_channels, columns, stride)
            self.fractal_blocks.append(block)
            in_channels = out_channels
            
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channels, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        
        # Forward through fractal blocks
        for block in self.fractal_blocks:
            outputs = block(x)
            x = join(outputs)  # Join block outputs
            
        # Global pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def fractalnet(num_classes=10, **kwargs):
    """Create FractalNet model"""
    return FractalNet(num_classes=num_classes, **kwargs)