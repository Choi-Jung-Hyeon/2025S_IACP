import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, 1, 1, bias=False)
        self.drop_rate = drop_rate
        
    def forward(self, x):
        new_features = self.conv1(self.relu(self.bn1(x)))
        new_features = self.conv2(self.relu(self.bn2(new_features)))
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module(f'denselayer{i+1}', layer)

class Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False))
        self.add_module('pool', nn.AvgPool2d(2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 12), 
                 num_init_features=16, bn_size=4, drop_rate=0, num_classes=10):
        super().__init__()
        
        # Initial convolution (CIFAR 논문 기준: 16 or 2*growth_rate)
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features, 3, 1, 1, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        
        # Dense blocks using ModuleList (6월 5일 comment)
        num_features = num_init_features
        block_layers = []
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate)
            block_layers.append(block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                # Transition layer with compression (논문: θ=0.5)
                trans = Transition(num_features, num_features // 2)
                block_layers.append(trans)
                num_features = num_features // 2
        
        self.block_layers = nn.ModuleList(block_layers)
        
        # Final batch norm
        self.final_bn = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        
        for layer in self.block_layers:
            x = layer(x)
            
        x = self.relu(self.final_bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _extract_features(self, x):
        x = self.features(x)
        for layer in self.block_layers:
            x = layer(x)
        x = self.relu(self.final_bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def densenet(num_classes=10, **kwargs):
    """DenseNet for CIFAR (논문 기준)"""
    if num_classes == 100:
        # CIFAR-100: DenseNet-BC (L=100, k=12)
        return DenseNet(growth_rate=12, block_config=(16, 16, 16), 
                       num_init_features=24, num_classes=num_classes, **kwargs)
    else:
        # CIFAR-10: DenseNet (L=40, k=12)
        return DenseNet(growth_rate=12, block_config=(6, 6, 6), 
                       num_init_features=16, num_classes=num_classes, **kwargs)