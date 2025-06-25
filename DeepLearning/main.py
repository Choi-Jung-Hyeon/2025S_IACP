import argparse
import os

import models
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

def main(args):
    # 모델 로드
    model = models.load_model(args.model, args.dataset)
    
    # 데이터셋 로드
    train_dataset = datasets.load_dataset(args.dataset, train=True)
    test_dataset = datasets.load_dataset(args.dataset, train=False)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # optimizer, criterion, scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                         momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # print_freq마다 출력
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Epoch: [{epoch+1}/{args.num_epochs}] '
                      f'Step: [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {train_loss/(batch_idx+1):.4f} '
                      f'Acc: {100.*correct/total:.2f}%')
        
        # eval_freq마다 평가
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            print(f'Epoch [{epoch+1}] Test Loss: {test_loss/len(test_loader):.4f} '
                  f'Test Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
    
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet34", 
                       help="Model: resnet34, densenet, fractalnet, preactresnet")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                       help="Dataset: cifar10 or cifar100")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_step", type=int, default=30, help="LR scheduler step")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="LR scheduler gamma")
    parser.add_argument("--print_freq", type=int, default=50, help="print frequency")
    parser.add_argument("--eval_freq", type=int, default=1, help="evaluation frequency")
    
    args = parser.parse_args()
    main(args)
