import torch
import torch.nn as nn

class FractalNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FractalNet, self).__init__()
        # 간단한 convolutional network를 예시로 작성 (실제 FractalNet 구조는 복잡)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # (입력 224x224 -> 두번의 풀링 후 56x56)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x