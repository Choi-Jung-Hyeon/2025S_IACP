import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, 1, 1, bias=False)
    
    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f'layer{i}', DenseLayer(in_channels + i * growth_rate, growth_rate))

class Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False))
        self.add_module('pool', nn.AvgPool2d(2))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 6, 6), 
                 num_init_features=16, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )
        
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        self.features.add_module('final_bn', nn.BatchNorm2d(num_features))
        self.features.add_module('final_relu', nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def _extract_features(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        return torch.flatten(out, 1)

def densenet(num_classes=10, **kwargs):
    if num_classes == 100:
        return DenseNet(growth_rate=12, block_config=(16, 16, 16), 
                       num_init_features=24, num_classes=num_classes)
    else:
        return DenseNet(growth_rate=12, block_config=(6, 6, 6), 
                       num_init_features=16, num_classes=num_classes)