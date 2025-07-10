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
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class RotNet(nn.Module):
    """RotNet for rotation prediction"""
    def __init__(self, num_classes=4):
        super().__init__()
        
        # 초기 conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # conv blocks (NIN style)
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
        
        # global average pooling & classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(192, num_classes)
        
        # weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_rotation_labels=True):
        if return_rotation_labels:
            # rotation prediction mode
            with torch.no_grad():
                # create rotation labels
                batch_size = x.size(0)
                label = torch.zeros(batch_size * 4, dtype=torch.long, device=x.device)
                for i in range(4):
                    label[batch_size * i:batch_size * (i + 1)] = i
                
                # apply 4 rotations using torch.rot90
                arr = []
                for i in range(4):
                    arr.append(torch.rot90(x, k=i, dims=(2, 3)))
                arr = torch.cat(arr, dim=0)
            
            # forward pass
            features = self._extract_features(arr)
            output = self.out(features)
            
            return output, label
        else:
            # standard classification mode
            features = self._extract_features(x)
            output = self.out(features)
            return output
    
    def _extract_features(self, x):
        """Extract features from input"""
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # conv blocks
        x = self.block1(x)
        x = self.pool1(x)
        
        x = self.block2(x)
        x = self.pool2(x)
        
        x = self.block3(x)
        
        # global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

# factory function
def rotnet(num_classes=4):
    """Create RotNet model"""
    return RotNet(num_classes=num_classes)

# test
if __name__ == "__main__":
    model = rotnet(num_classes=4)
    print("RotNet:", model)
    
    x = torch.randn(2, 3, 32, 32)
    
    # rotation prediction mode
    print("\n=== Rotation Prediction Mode ===")
    output, labels = model(x, return_rotation_labels=True)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Labels:", labels)
    print("Labels shape:", labels.shape)
    
    # standard classification mode  
    print("\n=== Standard Classification Mode ===")
    output = model(x, return_rotation_labels=False)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)