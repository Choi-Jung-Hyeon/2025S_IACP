import torch
import torch.nn as nn

class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # BN -> ReLU -> Conv 순서
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                    stride=stride, bias=False)
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        shortcut = self.shortcut(out) if self.shortcut else x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + shortcut

class PreActResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 초기 conv (BN, ReLU 없음)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # PreAct blocks
        self.layer1 = nn.Sequential(
            PreActBlock(64, 64),
            PreActBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            PreActBlock(64, 128, stride=2),
            PreActBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            PreActBlock(128, 256, stride=2),
            PreActBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            PreActBlock(256, 512, stride=2),
            PreActBlock(512, 512)
        )
        
        # 마지막 BN, ReLU, Pool, FC
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# test
if __name__ == "__main__":
    model = PreActResNet(num_classes=10)
    print(model)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)