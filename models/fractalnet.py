import torch
import torch.nn as nn
import random

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
        
        # 가장 짧은 경로 (local path)
        self.local_path = ConvBlock(in_channels, out_channels, stride=stride)
        
        # 깊이가 1보다 클 때만 재귀적으로 두 개의 브랜치 생성
        if depth > 1:
            self.branch1 = FractalBlock(in_channels, out_channels, depth - 1, stride=stride)
            self.branch2 = FractalBlock(in_channels, out_channels, depth - 1, stride=stride)

    def forward(self, x):
        if self.depth == 1:
            return self.local_path(x)

        # 학습 시: Local Drop-path 적용
        if self.training:
            # 3개의 경로(local, branch1, branch2) 중 하나를 무작위로 선택
            path_choice = random.randint(0, 2)
            
            if path_choice == 0:
                return self.local_path(x)
            elif path_choice == 1:
                return self.branch1(x)
            else: # path_choice == 2
                return self.branch2(x)

        # 추론 시: 모든 경로의 출력을 평균
        else:
            out_local = self.local_path(x)
            out_rec1 = self.branch1(x)
            out_rec2 = self.branch2(x)
            
            # 모든 경로의 출력을 동일한 가중치로 평균
            return (out_local + out_rec1 + out_rec2) / 3.0

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
        
        # 풀링과 FC 레이어
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
