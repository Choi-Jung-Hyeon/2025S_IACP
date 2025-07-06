import torch
import torch.nn as nn

# Bottleneck layer
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, 
                              padding=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

# DenseBlock using nn.Sequential
class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        for i in range(num_layers):
            layer = Bottleneck(in_channels + i * growth_rate, growth_rate)
            self.add_module(f'denselayer{i+1}', layer)

# Transition layer
class Transition(nn.Module):
    def __init__(self, in_channels, compression=0.5):
        super().__init__()
        out_channels = int(in_channels * compression)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return self.pool(out)

# DenseNet
class DenseNet(nn.Module):
    def __init__(self, num_classes=10, growth_rate=32, block_layers=[6, 12, 24, 16], 
                 compression=0.5, num_init_features=64):
        super().__init__()
        
        # Initial convolution
        self.features = nn.Sequential()
        if num_classes == 100 or num_classes > 10:  # for CIFAR-100
            self.features.add_module('conv0', nn.Conv2d(3, num_init_features, 
                                                       kernel_size=3, stride=1, 
                                                       padding=1, bias=False))
        else:  # for CIFAR-10
            self.features.add_module('conv0', nn.Conv2d(3, num_init_features, 
                                                       kernel_size=3, stride=1, 
                                                       padding=1, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        
        # DenseBlocks and Transitions using ModuleList
        num_features = num_init_features
        for i, num_layers in enumerate(block_layers):
            # DenseBlock
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            
            # Transition layer (except for last block)
            if i != len(block_layers) - 1:
                trans = Transition(num_features, compression)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # Classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# test
if __name__ == "__main__":
    model = DenseNet(num_classes=100, block_layers=[6, 12, 24, 16])
    print(model)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
