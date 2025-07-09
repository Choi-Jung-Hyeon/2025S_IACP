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
            
            return (out_local + out_rec1 + out_rec2) / 3.0

class FractalNet(nn.Module):
    def __init__(self, num_classes=10, initial_channels=64, depth=3, columns=3):
        super().__init__()
        
        self.conv1 = ConvBlock(3, initial_channels, kernel_size=3, stride=1, padding=1)
        self.stage1 = self._make_stage(initial_channels, initial_channels, columns, depth)
        self.pool1 = nn.MaxPool2d(2) 
        self.stage2 = self._make_stage(initial_channels, initial_channels * 2, columns, depth)
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = self._make_stage(initial_channels * 2, initial_channels * 4, columns, depth)
        self.pool_final = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(initial_channels * 4, num_classes)
        
    def _make_stage(self, in_channels, out_channels, columns, depth):
        blocks = []
        # 첫 번째 블록은 채널 수를 변경할 수 있음
        blocks.append(FractalBlock(in_channels, out_channels, depth=depth))
        # 나머지 블록은 채널 수를 유지
        for _ in range(1, columns):
            blocks.append(FractalBlock(out_channels, out_channels, depth=depth))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.stage1(x)
        x = self.pool1(x)
        
        x = self.stage2(x)
        x = self.pool2(x)
        
        x = self.stage3(x)      
        x = self.pool_final(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
