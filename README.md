# 2025S_IACP
산학협력프로젝트 유형2 이한국교수님 팀 리포지토리

## 프로젝트 개요
Training-Free Detection of AI-Generated Images 주제로 진행되는 산학협력 프로젝트입니다.

## 주요 업데이트
1. **통합 main.py**: Supervised와 SSL 학습이 하나의 스크립트로 통합
2. **1-NN Evaluation**: SSL evaluation이 k-NN에서 1-NN으로 변경
3. **Test Loss Tracking**: 10 epoch마다 test loss 출력
4. **PreActResNet 성능비교**: 기본 모델을 PreActResNet으로 설정

## 리포지토리 구조
```
main.py              # 통합 학습 스크립트
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
 ├─supervised_learning.py  # Base framework
 └─rotnet.py         # RotNet SSL + 1-NN evaluation
archive/             # 데이터셋
 ├─training_set/
 └─test_set/
```

## 실행방법

### Supervised Learning
```bash
python3 main.py --model MODEL --dataset DATASET
```

### Self-Supervised Learning
```bash
python3 main.py --framework rotnet --model MODEL --dataset DATASET
```

## 주요 Arguments

```
--model         모델 선택 (default: preactresnet)
                choices: resnet34, densenet, fractalnet, preactresnet, rotnet
--dataset       데이터셋 (default: cifar10)
                choices: cifar10, cifar100
--framework     SSL framework (default: None = supervised)
                choices: rotnet, None
--num_epochs    학습 epoch (default: 200, SSL: 100)
--batch_size    배치 크기 (default: 64, SSL: 128)
--lr            학습률 (default: 0.1)
--weight_decay  Weight decay (default: 5e-4)
--print_freq    출력 빈도 (default: 50)
--eval_freq     평가 빈도 (default: 1)
--num_blocks    RotNet NIN blocks (default: 4, choices: 3,4,5)
```

## 실행 예시

### 1. PreActResNet 성능 비교
```bash
# Supervised PreActResNet
python3 main.py --model preactresnet --dataset cifar10

# SSL RotNet with PreActResNet backbone
python3 main.py --framework rotnet --model preactresnet --dataset cifar10
```

### 2. 논문 검증 실행
```bash
# ResNet CIFAR
python3 main.py --model resnet34 --dataset cifar10

# DenseNet growth_rate=12
python3 main.py --model densenet --dataset cifar10

# FractalNet 40-layer
python3 main.py --model fractalnet --dataset cifar100

# RotNet NIN 구조
python3 main.py --framework rotnet --model rotnet --dataset cifar10
```

### 3. 성능 비교 실행
```bash
# 모든 모델 성능 비교
bash run_experiments.sh

# Supervised vs SSL 비교
python3 main.py --model preactresnet --dataset cifar10
python3 main.py --framework rotnet --model preactresnet --dataset cifar10
```

## 아키텍처 세부사항

### ResNet (CIFAR)
- **채널 구성**: {16, 32, 64} (논문 기준)
- **구조**: 6n+2 layers, 3-stage architecture
- **초기 conv**: 3×3, 16 channels

### PreActResNet (CIFAR)
- **Pre-activation**: BN → ReLU → Conv
- **채널 구성**: {16, 32, 64}
- **SSL 성능비교 기준 모델**

### DenseNet (CIFAR) 
- **Growth rate**: k=12 (논문 기준)
- **초기 채널**: 16 (CIFAR-10), 24 (CIFAR-100)
- **Block config**: (6,6,6) CIFAR-10, (16,16,16) CIFAR-100
- **Compression**: θ=0.5, Bottleneck: 4k features

### FractalNet (CIFAR)
- **구조**: B=5 blocks, C=4 columns, 40 layers
- **채널**: (64, 128, 256, 512, 512) (논문 기준)
- **학습**: LR=0.02, batch=100, 400 epochs (논문 기준)
- **Drop-path**: 50% local + 50% global sampling

### RotNet (NIN 구조)
- **ConvB1**: 96 × 16 × 16 feature maps
- **ConvB2-5**: 192 × 8 × 8 feature maps  
- **구조**: 각 block = 3×3 conv + 2×1×1 conv
- **학습**: LR=0.1, batch=128, 4개 회전 동시 처리

## SSL Framework 특징
- **RotNet framework**: 4개 회전 이미지 동시 처리
- **1-NN evaluation**: Fine-tuning 없이 L2 distance
- **Framework 상속**: supervised_learning.py 기반 모듈화
- **Test loss tracking**: 10 epoch마다 test loss 출력


# 자주 쓰는 명령어
```
Ctrl+b → %               tmux: 오른쪽에 새 창 생성
Ctrl+b → "               tmux: 아래에 새 창 생성
Ctrl+b → (방향키)        tmux: 방향키 방향의 창으로 이동
Ctrl+d                   tmux: 현재 창 닫기
watch -n 0.1 nvidia-smi  GPU 사용량을 0.1초마다 모니터링
```
