import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, stride=1):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(0.5)
        
        self.block = ConvBlock(in_channels, out_channels, stride=stride)
        if depth > 1:
            # 두 브랜치가 병렬로 존재하도록 수정
            self.branch1 = FractalBlock(in_channels, out_channels, depth - 1, stride=stride)
            self.branch2 = FractalBlock(in_channels, out_channels, depth - 1, stride=stride)
    
    def forward(self, x):
        if self.depth == 1:
            return self.block(x)
        
        # 로컬 경로 (가장 짧은 경로)
        out_local = self.block(x)
        out_local = self.dropout(out_local)
        
        # 재귀적 경로 (병렬 처리)
        out_rec1 = self.branch1(x)
        out_rec2 = self.branch2(x)
        
        # 재귀 경로의 출력을 평균
        out_recursive = (out_rec1 + out_rec2) / 2.0
        out_recursive = self.dropout(out_recursive)

        # 로컬 경로와 재귀 경로의 출력을 평균내어 최종 출력 결정
        return (out_local + out_recursive) / 2.0

class FractalNet(nn.Module):
    def __init__(self, num_classes=10, num_columns=5, depth=4):
        super().__init__()
        self.num_columns = num_columns
        self.depth = depth
        
        self.conv1 = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1)
        
        # Fractal 블록들을 담을 ModuleList
        self.blocks = nn.ModuleList()
        for _ in range(self.num_columns):
            self.blocks.append(FractalBlock(64, 64, depth, stride=1))
        
        # 풀링과 FC 레이어는 forward에서 한 번만 호출
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        
        for block in self.blocks:
            x = block(x)
        
        # 모든 블록 통과 후 풀링 및 FC 레이어 적용
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
