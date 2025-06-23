import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Fractal block with proper branching
class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, stride=1):
        super().__init__()
        self.num_columns = num_columns
        self.columns = nn.ModuleList()
        
        # 각 column은 2^c개의 conv blocks를 가짐
        for c in range(num_columns):
            column = nn.ModuleList()
            for _ in range(2**c):
                if stride == 2 and len(column) == 0:
                    # 첫 번째 블록에서만 stride 적용
                    column.append(ConvBlock(in_channels, out_channels, stride=2))
                else:
                    column.append(ConvBlock(out_channels if len(column) > 0 else in_channels, 
                                          out_channels))
            self.columns.append(column)
    
    def forward(self, x, active_columns=None):
        if active_columns is None:
            active_columns = list(range(self.num_columns))
        
        results = []
        for c in active_columns:
            out = x
            for block in self.columns[c]:
                out = block(out)
            results.append(out)
        
        # 평균 (입력 개수로 나눔)
        if len(results) == 1:
            return results[0]
        else:
            return sum(results) / len(results)

# Main FractalNet
class FractalNet(nn.Module):
    def __init__(self, num_classes=10, num_columns=4, channels=[64, 128, 256, 512]):
        super().__init__()
        self.num_columns = num_columns
        
        # Initial conv
        self.conv1 = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1)
        
        # Fractal blocks (여러 개 사용)
        self.blocks = nn.ModuleList()
        in_channels = 64
        for out_channels in channels:
            if in_channels == 64:
                self.blocks.append(FractalBlock(in_channels, out_channels, num_columns, stride=1))
            else:
                self.blocks.append(FractalBlock(in_channels, out_channels, num_columns, stride=2))
            in_channels = out_channels
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.conv1(x)
        
        # Forward through fractal blocks
        for block in self.blocks:
            # Training: randomly select columns (drop-path)
            if self.training:
                # 최소 1개의 column은 활성화
                num_active = torch.randint(1, self.num_columns + 1, (1,)).item()
                active_columns = torch.randperm(self.num_columns)[:num_active].tolist()
                x = block(x, active_columns)
            else:
                # Test: use all columns
                x = block(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# test
if __name__ == "__main__":
    model = FractalNet(num_classes=100, num_columns=4, channels=[64, 128, 256, 512])
    print(model)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)