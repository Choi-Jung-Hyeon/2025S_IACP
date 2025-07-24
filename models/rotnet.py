import torch
import torch.nn as nn

class NINBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.layers(x)

class RotNetModel(nn.Module):
    def __init__(self, num_classes=4, num_blocks=4):
        super().__init__()
        channels = [96] + [192] * (num_blocks - 1)
        
        layers = [NINBlock(3, channels[0])]
        if num_blocks > 1:
            layers.append(nn.MaxPool2d(2, 2))
            layers.append(NINBlock(channels[0], channels[1]))
            
            if num_blocks > 2:
                layers.append(nn.MaxPool2d(2, 2))
                for i in range(2, num_blocks):
                    layers.append(NINBlock(channels[i-1], channels[i]))
        
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _extract_features(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

def rotnet(num_classes=4, num_blocks=4):
    return RotNetModel(num_classes, num_blocks)