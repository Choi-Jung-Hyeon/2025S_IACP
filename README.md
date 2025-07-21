# 2025S_IACP
산학협력프로젝트 유형2 이한국교수님 팀 리포지토리

## 프로젝트 개요
Training-Free Detection of AI-Generated Images 주제로 진행되는 산학협력 프로젝트입니다.

## 리포지토리 구조
```
main.py              # Supervised learning 스크립트
main_ssl.py          # Self-supervised learning 스크립트  
datasets.py          # 데이터셋 로딩 및 augmentation
models/              # Encoder architectures
 ├─__init__.py
 ├─ResNet.py         # ResNet CIFAR {16,32,64}
 ├─PreActResNet.py   # Pre-activation ResNet CIFAR
 ├─DenseNet.py       # DenseNet k=12, CIFAR 최적화
 ├─FractalNet.py     # FractalNet B=5,C=4
 └─rotnet.py         # Network-in-Network 96→192
frameworks/          # Learning frameworks
 ├─__init__.py
 ├─supervised_learning.py
 └─rotnet.py         # RotNet SSL + k-NN evaluation
archive/             # 데이터셋 (강아지/고양이 등)
 ├─training_set/
 └─test_set/
```


### CIFAR 전용 아키텍처

#### ResNet (CIFAR)
- **채널 구성**: {16, 32, 64} (논문 기준)
- **구조**: 6n+2 layers, 3-stage architecture
- **초기 conv**: 3×3, 16 channels

#### DenseNet (CIFAR) 
- **Growth rate**: k=12 (논문 기준)
- **초기 채널**: 16 (CIFAR-10), 24 (CIFAR-100)
- **Block config**: (6,6,6) CIFAR-10, (16,16,16) CIFAR-100
- **Compression**: θ=0.5, Bottleneck: 4k features

#### FractalNet (CIFAR)
- **구조**: B=5 blocks, C=4 columns, 40 layers
- **채널**: (64, 128, 256, 512, 512) (논문 기준)
- **학습**: LR=0.02, batch=100, 400 epochs (논문 기준)
- **Drop-path**: 50% local + 50% global sampling
- depth-length tensor list 반환

#### RotNet (NIN 구조)
- **ConvB1**: 96 × 16 × 16 feature maps
- **ConvB2-5**: 192 × 8 × 8 feature maps  
- **구조**: 각 block = 3×3 conv + 2×1×1 conv
- **학습**: LR=0.1, batch=128, 4개 회전 동시 처리

### SSL Framework
- **RotNet framework**: 4개 회전 이미지 동시 처리
- **k-NN evaluation**: Fine-tuning 없이 L2/cosine distance
- **Framework 구조**: encoder → SSL loss, extract_features 분리

## 실행방법

### Supervised Learning
```bash
python3 main.py --model MODEL --dataset DATASET
```

### Self-Supervised Learning
```bash
python3 main_ssl.py --framework rotnet --model MODEL --dataset DATASET
```

## 주요 Arguments

### main.py
```
--model         resnet34, densenet, fractalnet, preactresnet
--dataset       cifar10, cifar100
--num_epochs    (default: 200, fractalnet: 400)
--batch_size    (default: 64, fractalnet: 100)
--lr            (default: 0.1, fractalnet: 0.02)
--weight_decay  (default: 5e-4)
--lr_step       (default: 100, fractalnet: 200)
--lr_gamma      (default: 0.1)
--print_freq    (default: 50)
--eval_freq     (default: 1)
```

### main_ssl.py
```
--framework     rotnet
--model         rotnet, resnet34, densenet, fractalnet, preactresnet
--dataset       cifar10, cifar100
--ssl_epochs    (default: 100)
--batch_size    (default: 128)
--lr            (default: 0.1)
--weight_decay  (default: 5e-4)
--num_blocks    (default: 4, choices: 3,4,5)
```

## 실행 예시

### 논문 검증 실행
```bash
# ResNet CIFAR {16,32,64} 채널 검증
python3 main.py --model resnet34 --dataset cifar10

# DenseNet growth_rate=12 검증  
python3 main.py --model densenet --dataset cifar10

# FractalNet 40-layer (B=5,C=4) 검증 + 논문 학습 설정
python3 main.py --model fractalnet --dataset cifar100

# PreActResNet CIFAR 구조 검증
python3 main.py --model preactresnet --dataset cifar10

# RotNet NIN 구조 검증 (96→192 채널)
python3 main_ssl.py --framework rotnet --model rotnet --dataset cifar10

# RotNet with 5 NIN blocks
python3 main_ssl.py --framework rotnet --model rotnet --num_blocks 5

# ResNet backbone으로 RotNet SSL
python3 main_ssl.py --framework rotnet --model resnet34 --dataset cifar10
```

### 성능 비교 실행
```bash
# DenseNet vs FractalNet vs ResNet 성능 비교
python3 main.py --model densenet --dataset cifar100 --num_epochs 300
python3 main.py --model fractalnet --dataset cifar100  # 자동 400 epochs
python3 main.py --model resnet34 --dataset cifar100 --num_epochs 300

# RotNet SSL vs Supervised 비교
python3 main_ssl.py --framework rotnet --model rotnet --dataset cifar10
python3 main.py --model rotnet --dataset cifar10
```