# densenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
#     Model Components
# -------------------------

class Bottleneck(nn.Module):
    """
    DenseNet Block 내 하나의 Bottleneck Layer.
    - 먼저 1x1 Conv를 통해 채널을 4배 확장하고,
    - 이후 3x3 Conv를 통해 growth_rate(k) 개의 feature를 추출한 후,
    - 입력과 결과를 concatenate (채널 축으로 결합)
    """
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inner_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(inner_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inner_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        # Concatenate along channels dimension
        out = torch.cat([x, out], dim=1)
        return out

class DenseBlock(nn.Module):
    """
    여러 Bottleneck Layer를 순차적으로 쌓은 Dense Block.
    각 레이어마다 입력 채널 수가 증가합니다.
    """
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class Transition(nn.Module):
    """
    Dense Block 사이의 Transition Layer.
    1×1 Conv를 통해 채널 수를 압축(compression)한 후,
    평균 풀링을 적용하여 공간 해상도를 절반으로 줄입니다.
    """
    def __init__(self, in_channels, compression):
        super(Transition, self).__init__()
        out_channels = int(in_channels * compression)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out

# -------------------------
#        DenseNet
# -------------------------
class DenseNet(nn.Module):
    """
    PyTorch 구현 DenseNet.
    구성:
      - Initial Conv: 7x7 conv, stride=2 — (입력 224x224, output 112x112)
      - MaxPool: 3x3, stride=2  -> 56x56
      - DenseBlock1 (6 layers) -> Transition (reduce resolution to 28x28)
      - DenseBlock2 (6 layers) -> Transition (14x14)
      - DenseBlock3 (6 layers) -> Transition (7x7)
      - DenseBlock4 (6 layers)
      - Global Average Pool & FC layer for classification (classes=2)
    
    하이퍼파라미터:
      - growth_rate: 각 Bottleneck에서 생성하는 feature map 수 (기본 32)
      - compression: Transition Layer에서 채널 압축 비율 (기본 0.5)
      - block_layers: 각 Dense Block 내 레이어 수 (여기서는 [6, 6, 6, 6])
    """
    def __init__(self, num_classes=2, growth_rate=32, compression=0.5, block_layers=[6, 6, 6, 6]):
        super(DenseNet, self).__init__()
        num_init_features = growth_rate * 2  # 2*k, 즉 64 channels
        
        # Initial convolution and pooling
        self.conv0 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(num_init_features)
        self.relu = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Output: 56x56
        
        # Dense Block 1
        num_features = num_init_features
        self.denseblock1 = DenseBlock(block_layers[0], num_features, growth_rate)
        num_features = num_features + block_layers[0] * growth_rate  # 예: 64 + 6*32 = 256
        self.trans1 = Transition(num_features, compression)
        num_features = int(num_features * compression)  # 예: 256*0.5 = 128
        
        # Dense Block 2
        self.denseblock2 = DenseBlock(block_layers[1], num_features, growth_rate)
        num_features = num_features + block_layers[1] * growth_rate  # 128 + 6*32 = 320
        self.trans2 = Transition(num_features, compression)
        num_features = int(num_features * compression)  # 320*0.5 = 160
        
        # Dense Block 3
        self.denseblock3 = DenseBlock(block_layers[2], num_features, growth_rate)
        num_features = num_features + block_layers[2] * growth_rate  # 160 + 6*32 = 352
        self.trans3 = Transition(num_features, compression)
        num_features = int(num_features * compression)  # 352*0.5 = 176
        
        # Dense Block 4
        self.denseblock4 = DenseBlock(block_layers[3], num_features, growth_rate)
        num_features = num_features + block_layers[3] * growth_rate  # 176 + 6*32 = 368
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.pool0(x)
        
        x = self.denseblock1(x)
        x = self.trans1(x)
        x = self.denseblock2(x)
        x = self.trans2(x)
        x = self.denseblock3(x)
        x = self.trans3(x)
        x = self.denseblock4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -------------------------
#       Training Code
# -------------------------
if __name__ == "__main__":
    # PyTorch는 GPU 사용을 위한 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # Dataset: Kaggle Dog & Cat Dataset
    # 예시에서는 ImageFolder로 지정 (폴더 내에 클래스별 디렉토리가 있어야 함)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset_path = os.path.join("./archive")
    train_dataset_path = os.path.join(dataset_path, "train_set")
    valid_dataset_path = os.path.join(dataset_path, "validation_set")
    
    print("Path to train dataset: ", train_dataset_path)
    print("Path to validation dataset: ", valid_dataset_path)
    
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform)
    valid_dataset = datasets.ImageFolder(root=valid_dataset_path, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 모델 생성
    model = DenseNet(num_classes=2).to(device)
    
    # Optimizer and Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total_train += labels.size(0)
            correct_train += preds.eq(labels).sum().item()
        
        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        
        # Validation
        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss_val += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                total_val += labels.size(0)
                correct_val += preds.eq(labels).sum().item()
        
        epoch_loss_val = running_loss_val / total_val
        epoch_acc_val = correct_val / total_val
        val_loss_history.append(epoch_loss_val)
        val_acc_history.append(epoch_acc_val)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {epoch_loss_val:.4f}, Val Acc: {epoch_acc_val:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), 'densenet_model.pth')
    
    # Plot Accuracy and Loss Graphs
    plt.figure(1)
    plt.plot(train_acc_history, label='train')
    plt.plot(val_acc_history, label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.savefig('DenseNet_Accuracy.png')
    
    plt.figure(2)
    plt.plot(train_loss_history, label='train')
    plt.plot(val_loss_history, label='validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.savefig('DenseNet_Loss.png')
