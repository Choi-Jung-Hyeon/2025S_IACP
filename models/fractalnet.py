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
    def __init__(self, in_channels, out_channels, num_columns, stride=1):
        super().__init__()
        self.num_columns = num_columns
        self.stride = stride
        
        # 각 column을 별도의 path로 생성
        self.columns = nn.ModuleList()
        for c in range(num_columns):
            # c번째 column은 2^c개의 conv block을 가짐
            column_depth = 2 ** c
            column_blocks = nn.ModuleList()
            
            for d in range(column_depth):
                if d == 0 and stride == 2:
                    # 첫 번째 block에서만 stride 적용
                    block = ConvBlock(in_channels, out_channels, stride=stride)
                else:
                    # 나머지는 모두 동일한 채널로
                    block = ConvBlock(out_channels if d > 0 else in_channels, out_channels)
                column_blocks.append(block)
            
            self.columns.append(column_blocks)
    
    def forward(self, x, drop_path_enabled=False, active_columns=None):
        if drop_path_enabled and self.training:
            # Local drop-path: 랜덤하게 일부 column 선택
            if active_columns is None:
                num_active = torch.randint(1, self.num_columns + 1, (1,)).item()
                active_columns = torch.randperm(self.num_columns)[:num_active].tolist()
        else:
            # 모든 column 사용
            active_columns = list(range(self.num_columns))
        
        outputs = []
        for c in active_columns:
            out = x
            # 각 column의 모든 block을 순차적으로 통과
            for block in self.columns[c]:
                out = block(out)
            outputs.append(out)
        
        # Join: element-wise mean (활성화된 column 수로 나눔)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return torch.stack(outputs).mean(dim=0)

class FractalNet(nn.Module):
    def __init__(self, num_classes=10, num_columns=4, channels=[64, 128, 256, 512], 
                 drop_path_rate=0.15):
        super().__init__()
        self.num_columns = num_columns
        self.drop_path_rate = drop_path_rate
        
        # Initial convolution
        self.conv1 = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1)
        
        # 여러 FractalBlock들을 생성
        self.fractal_blocks = nn.ModuleList()
        in_channels = 64
        
        for i, out_channels in enumerate(channels):
            # 첫 번째 block을 제외하고는 stride=2로 downsampling
            stride = 1 if i == 0 else 2
            block = FractalBlock(in_channels, out_channels, num_columns, stride)
            self.fractal_blocks.append(block)
            in_channels = out_channels
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(channels[-1], num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, global_drop_path=False):
        x = self.conv1(x)
        
        # Global drop-path: 전체 네트워크에서 하나의 column만 선택
        if global_drop_path and self.training:
            selected_column = torch.randint(0, self.num_columns, (1,)).item()
            active_columns = [selected_column]
        else:
            active_columns = None
        
        # 모든 FractalBlock을 통과
        for block in self.fractal_blocks:
            # Local drop-path 확률적 적용
            use_drop_path = self.training and torch.rand(1).item() < self.drop_path_rate
            
            if global_drop_path and self.training:
                # Global drop-path 모드
                x = block(x, drop_path_enabled=False, active_columns=active_columns)
            else:
                # Local drop-path 모드
                x = block(x, drop_path_enabled=use_drop_path)
        
        # Classification head
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# CIFAR용 설정
def fractalnet_cifar(num_classes=10, num_columns=4):
    return FractalNet(
        num_classes=num_classes,
        num_columns=num_columns,
        channels=[64, 128, 256, 512],
        drop_path_rate=0.15
    )

# 테스트
if __name__ == "__main__":
    model = fractalnet_cifar(num_classes=100, num_columns=4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    
    # Normal forward
    y1 = model(x)
    print(f"Normal output shape: {y1.shape}")
    
    # Global drop-path forward
    model.train()
    y2 = model(x, global_drop_path=True)
    print(f"Global drop-path output shape: {y2.shape}")
    
    # 각 column의 depth 출력
    for i, block in enumerate(model.fractal_blocks):
        print(f"Block {i+1} columns depths: {[len(col) for col in block.columns]}")
