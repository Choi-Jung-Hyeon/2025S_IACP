# 2025S_IACP
산학협력프로젝트 유형2 이한국교수님 팀 리포지토리

# 실행방법
```
python3 main.py (argument)
```
## Argument
```
--model         (default: resnet18), resnet34, densenet, fractalnet, preactresnet
--dataset       (default: cifar10), cifar100
--num_epochs    (default: 100) 학습을 진행할 Epoch 수
--batch_size    (default: 64) 학습에 사용할 Batch Size
--lr            (default: 0.1) 초기 Learning Rate
--weight_decay  (default: 5e-4) Weight Decay
--lr_step       (dafault: 30) Learning Rate Schedular Step
--lr_gamma      (default: 0.1) Learning Rate Schedular Gamma
--print_freq    (default: 50) 학습 도중 Training Accuracy를 출력할 Frequency
--eval_freq     (default: 1) Epoch별로 Validate Accuracy를 출력할 Frequency
```

# 리포지토리 구조
```
main.py
datasets.py
models/
 └__init__.py
 └ResNet.py
 └PreActResNet.py
 └DenseNet.py
 └FractalNet.py
```
