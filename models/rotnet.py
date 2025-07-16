import torch
import torch.nn as nn

class RotNetBlock(nn.Module):
    """RotNet basic conv block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class RotNet(nn.Module):
    """RotNet architecture for rotation prediction"""
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Conv blocks (Network-in-Network style)
        self.block1 = nn.Sequential(
            RotNetBlock(64, 96),
            RotNetBlock(96, 96),
            RotNetBlock(96, 96)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block2 = nn.Sequential(
            RotNetBlock(96, 192),
            RotNetBlock(192, 192),
            RotNetBlock(192, 192)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block3 = nn.Sequential(
            RotNetBlock(192, 192),
            RotNetBlock(192, 192),
            RotNetBlock(192, 192)
        )
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, num_classes)
        
    def forward(self, x, return_rotation_labels=False):
        """Forward pass with optional rotation label generation"""
        if return_rotation_labels:
            # Generate 4 rotated versions
            batch_size = x.size(0)
            
            x_rot0 = x  # 0°
            x_rot1 = torch.rot90(x, 1, dims=[2, 3])  # 90°
            x_rot2 = torch.rot90(x, 2, dims=[2, 3])  # 180°
            x_rot3 = torch.rot90(x, 3, dims=[2, 3])  # 270°
            
            x_all = torch.cat([x_rot0, x_rot1, x_rot2, x_rot3], dim=0)
            
            # Create rotation labels
            labels = torch.cat([
                torch.zeros(batch_size, dtype=torch.long),
                torch.ones(batch_size, dtype=torch.long),
                torch.full((batch_size,), 2, dtype=torch.long),
                torch.full((batch_size,), 3, dtype=torch.long)
            ], dim=0).to(x.device)
            
            # Extract features and classify
            features = self._extract_features(x_all)
            output = self.fc(features)
            
            return output, labels
        else:
            # Standard forward pass
            features = self._extract_features(x)
            output = self.fc(features)
            return output
    
    def _extract_features(self, x):
        """Extract features from input"""
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Conv blocks
        x = self.block1(x)
        x = self.pool1(x)
        
        x = self.block2(x)
        x = self.pool2(x)
        
        x = self.block3(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

def rotnet(num_classes=4):
    """Create RotNet model"""
    return RotNet(num_classes=num_classes)