import torch
import torch.nn as nn
import random

def join(inputs):
    """Join multiple tensors by element-wise mean"""
    return torch.stack(inputs).mean(dim=0)

class ConvBlock(nn.Module):
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
            # Two separate fractal sub-blocks
            self.block1 = FractalBlock(in_channels, out_channels, depth-1, stride, drop_path)
            self.block2 = FractalBlock(out_channels, out_channels, depth-1, 1, drop_path)
            
    def forward(self, x):
        # Base path output
        y = [self.block0(x)]
        
        if self.depth > 1:
            # Left branch
            branch1_outputs = self.block1(x)
            z = join(branch1_outputs)
            
            # Right branch
            branch2_outputs = self.block2(z)
            
            # Extend with branch outputs (y.extend, not y.expand)
            y.extend(branch2_outputs)
            
        # Apply drop-path during training
        if self.training and self.drop_path > 0:
            y = self._apply_drop_path(y)
            
        return y  # Returns depth-length tensor list
    
    def _apply_drop_path(self, outputs):
        """Drop-path regularization (논문: 50% local + 50% global)"""
        if len(outputs) == 1:
            return outputs
            
        if random.random() < 0.5:
            # Local sampling
            num_keep = max(1, len(outputs) // 2)
            indices = random.sample(range(len(outputs)), num_keep)
            return [outputs[i] for i in sorted(indices)]
        else:
            # Global sampling (single column)
            idx = random.randint(0, len(outputs) - 1)
            return [outputs[idx]]

class FractalNet(nn.Module):
    """FractalNet"""
    def __init__(self, num_classes=10, columns=4, blocks=5, drop_path=0.15):
        super().__init__()
        self.columns = columns
        self.blocks = blocks
        
        # Initial convolution
        self.conv1 = ConvBlock(3, 64)
        
        # Fractal blocks (논문: 64, 128, 256, 512, 512)
        self.fractal_blocks = nn.ModuleList()
        channel_config = [64, 128, 256, 512, 512]
        
        for i in range(blocks):
            in_channels = channel_config[i-1] if i > 0 else 64
            out_channels = channel_config[i]
            stride = 2 if i > 0 else 1  # 논문: 2x2 pooling after each block
                
            block = FractalBlock(in_channels, out_channels, columns, stride, drop_path)
            self.fractal_blocks.append(block)
            
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        
        # Forward through fractal blocks (join 처리)
        for block in self.fractal_blocks:
            outputs = block(x)
            x = join(outputs)  # Join block outputs
            
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def _extract_features(self, x):
        x = self.conv1(x)
        for block in self.fractal_blocks:
            outputs = block(x)
            x = join(outputs)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

def fractalnet(num_classes=10, **kwargs):
    """FractalNet (B=5, C=4, 40-layer)"""
    return FractalNet(num_classes=num_classes, columns=4, blocks=5, **kwargs)