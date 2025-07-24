import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import models
import datasets

def main(args):
    # 모델 및 데이터셋 로드
    num_classes = datasets.get_num_classes(args.dataset)
    model = models.load_model(args.model, num_classes=num_classes)
    train_dataset = datasets.load_dataset(args.dataset, train=True, ssl_mode=False)
    test_dataset = datasets.load_dataset(args.dataset, train=False, ssl_mode=False)

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Optimizer 선택
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    criterion = nn.CrossEntropyLoss()
    # 스케줄러 설정
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    print("=" * 70)
    print("SUPERVISED LEARNING")
    print("=" * 70)
    print(f"Model: {args.model}, Dataset: {args.dataset}, Optimizer: {args.optimizer.upper()}")
    print(f"Epochs: {args.num_epochs}, Batch Size: {args.batch_size}, Learning Rate: {args.lr}")
    print("=" * 70)
    
    # 정확도 저장을 위한 리스트
    train_accuracies = []
    test_accuracies = []

    # 학습 루프
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Epoch: [{epoch+1}/{args.num_epochs}] | Step: [{batch_idx+1}/{len(train_loader)}] | Loss: {train_loss/(batch_idx+1):.4f} | Acc: {100.*correct_train/total_train:.2f}%')
        
        # 에포크 학습 정확도 기록
        train_accuracies.append(100. * correct_train / total_train)
        
        # 평가
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            test_loss, correct_test, total_test = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_test += labels.size(0)
                    correct_test += predicted.eq(labels).sum().item()
            
            current_test_acc = 100. * correct_test / total_test
            test_accuracies.append(current_test_acc)
            print(f'Epoch [{epoch+1}] Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {current_test_acc:.2f}%')
        else:
            # 평가를 건너뛴 에포크는 이전 정확도 값을 그대로 사용
            if test_accuracies:
                test_accuracies.append(test_accuracies[-1])
            else:
                test_accuracies.append(0)
        
        scheduler.step()
    
    print("=" * 70)
    print("SUPERVISED LEARNING FINISHED")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Supervised Learning')
    
    # 기본 인자
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    
    # Optimizer 선택 인자
    parser.add_argument("--optimizer", type=str, default="sgd")
    
    # 학습 하이퍼파라미터
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--lr_step", type=int, default=30, help="LR scheduler step size")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="LR scheduler gamma")
    
    # 로그 및 평가 주기
    parser.add_argument("--print_freq", type=int, default=50, help="Print frequency")
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluation frequency")
    
    args = parser.parse_args()
    main(args)
