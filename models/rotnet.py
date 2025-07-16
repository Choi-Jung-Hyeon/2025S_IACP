import torch
import torch.nn as nn

class NINBlock(nn.Module):
    """Network-in-Network block (3 conv layers)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        
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
    """RotNet with NIN architecture for CIFAR-10 (논문 기반)"""
    def __init__(self, num_classes=4, num_blocks=4):
        super().__init__()
        self.num_blocks = num_blocks
        
        # NIN blocks
        self.blocks = nn.ModuleList()
        
        # Block 1: 3 -> 96, feature map: 96 × 16 × 16
        self.blocks.append(NINBlock(3, 96, kernel_size=3, stride=1, padding=1))
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 32x32 -> 16x16
        
        # Block 2: 96 -> 192, feature map: 192 × 8 × 8
        self.blocks.append(NINBlock(96, 192, kernel_size=3, stride=1, padding=1))
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 16x16 -> 8x8
        
        # Block 3: 192 -> 192 (optional)
        if num_blocks >= 3:
            self.blocks.append(NINBlock(192, 192, kernel_size=3, stride=1, padding=1))
            
        # Block 4: 192 -> 192 (optional)
        if num_blocks >= 4:
            self.blocks.append(NINBlock(192, 192, kernel_size=3, stride=1, padding=1))
            
        # Block 5: 192 -> 192 (optional)
        if num_blocks >= 5:
            self.blocks.append(NINBlock(192, 192, kernel_size=3, stride=1, padding=1))
        
        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(192, num_classes)
        
    def forward(self, x, return_rotation_labels=False):
        """Forward pass with optional rotation label generation"""
        if return_rotation_labels:
            # Generate 4 rotated versions (논문 방식)
            batch_size = x.size(0)
            
            x_rot0 = x  # 0°
            x_rot1 = torch.rot90(x, 1, dims=[2, 3])  # 90°
            x_rot2 = torch.rot90(x, 2, dims=[2, 3])  # 180°
            x_rot3 = torch.rot90(x, 3, dims=[2, 3])  # 270°
            
            # Stack all rotations (feed simultaneously)
            x_all = torch.cat([x_rot0, x_rot1, x_rot2, x_rot3], dim=0)
            
            # Create rotation labels
            labels = torch.cat([
                torch.zeros(batch_size, dtype=torch.long),
                torch.ones(batch_size, dtype=torch.long),
                torch.full((batch_size,), 2, dtype=torch.long),
                torch.full((batch_size,), 3, dtype=torch.long)
            ], dim=0).to(x.device)
            
            # Forward pass
            features = self._extract_features(x_all)
            output = self.classifier(features)
            
            return output, labels
        else:
            # Standard forward pass
            features = self._extract_features(x)
            output = self.classifier(features)
            return output
    
    def _extract_features(self, x):
        """Extract features from input"""
        # Block 1
        x = self.blocks[0](x)
        x = self.pool1(x)
        
        # Block 2
        x = self.blocks[1](x)
        x = self.pool2(x)
        
        # Additional blocks
        for i in range(2, len(self.blocks)):
            x = self.blocks[i](x)
        
        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_feature_maps(self, x, block_idx):
        """Get feature maps from specific block (for evaluation)"""
        if block_idx == 0:
            return self.blocks[0](x)
        elif block_idx == 1:
            x = self.blocks[0](x)
            x = self.pool1(x)
            return self.blocks[1](x)
        else:
            x = self.blocks[0](x)
            x = self.pool1(x)
            x = self.blocks[1](x)
            x = self.pool2(x)
            for i in range(2, min(block_idx + 1, len(self.blocks))):
                x = self.blocks[i](x)
            return x

def rotnet(num_classes=4, num_blocks=4):
    """Create RotNet model (논문 기반 NIN 구조)"""
    return RotNet(num_classes=num_classes, num_blocks=num_blocks)