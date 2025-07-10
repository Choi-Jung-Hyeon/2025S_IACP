import argparse
import os

import models
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

def get_model_specific_args(parser):
    """각 모델별로 특화된 argument 추가"""
    # DenseNet specific
    parser.add_argument("--growth_rate", type=int, default=32, 
                       help="DenseNet growth rate")
    parser.add_argument("--block_layers", type=int, nargs='+', default=[6, 12, 24, 16],
                       help="DenseNet block layers (e.g., --block_layers 6 12 24 16)")
    parser.add_argument("--compression", type=float, default=0.5,
                       help="DenseNet compression ratio")
    
    # ResNet specific  
    parser.add_argument("--resnet_type", type=str, default="resnet34",
                       help="ResNet type: resnet18, resnet34, resnet50")
    
    # SSL specific
    parser.add_argument("--ssl_mode", action="store_true",
                       help="Use Self-Supervised Learning (RotNet)")
    parser.add_argument("--ssl_pretrain_epochs", type=int, default=100,
                       help="SSL pretraining epochs")
    parser.add_argument("--ssl_finetune_epochs", type=int, default=50,  
                       help="SSL finetuning epochs")
    parser.add_argument("--ssl_freeze_backbone", action="store_true",
                       help="Freeze backbone during SSL finetuning")

def load_model_with_args(args):
    """args에 따라 모델별 파라미터를 전달하여 모델 로드"""
    model_name = args.model.lower()
    
    if model_name == "densenet":
        return models.load_model(model_name, 
                               num_classes=args.num_classes,
                               growth_rate=args.growth_rate,
                               block_layers=args.block_layers,
                               compression=args.compression)
    elif model_name == "rotnet":
        if args.ssl_mode:
            return models.load_model(model_name, num_classes=4)  # 4 rotations
        else:
            return models.load_model(model_name, num_classes=args.num_classes)
    else:
        return models.load_model(model_name, num_classes=args.num_classes)

def ssl_pretrain(model, train_loader, device, args):
    """Self-Supervised Learning 사전훈련"""
    print("=== SSL Pretraining Phase ===")
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                         momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    model.train()
    for epoch in range(args.ssl_pretrain_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            
            # RotNet forward: rotation prediction mode
            outputs, rot_labels = model(inputs, return_rotation_labels=True)
            rot_labels = rot_labels.to(device)
            
            loss = criterion(outputs, rot_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += rot_labels.size(0)
            correct += predicted.eq(rot_labels).sum().item()
            
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'SSL Epoch: [{epoch+1}/{args.ssl_pretrain_epochs}] '
                      f'Step: [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {running_loss/(batch_idx+1):.4f} '
                      f'Rot_Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        
        if (epoch + 1) % args.eval_freq == 0:
            print(f'SSL Epoch [{epoch+1}] Train Loss: {running_loss/len(train_loader):.4f} '
                  f'Rotation Acc: {100.*correct/total:.2f}%')

def ssl_finetune(model, train_loader, test_loader, device, args):
    """SSL 사전훈련된 모델을 classification에 맞게 finetuning"""
    print("=== SSL Finetuning Phase ===")
    
    # classification head 교체 (4 -> num_classes)
    model.out = nn.Linear(model.out.in_features, args.num_classes).to(device)
    
    # backbone freeze 옵션
    if args.ssl_freeze_backbone:
        for name, param in model.named_parameters():
            if 'out' not in name:  # classifier만 학습
                param.requires_grad = False
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr * 0.1,  # lower lr for finetuning
                         momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    for epoch in range(args.ssl_finetune_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # standard classification mode
            outputs = model(inputs, return_rotation_labels=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Finetune Epoch: [{epoch+1}/{args.ssl_finetune_epochs}] '
                      f'Step: [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {train_loss/(batch_idx+1):.4f} '
                      f'Acc: {100.*correct/total:.2f}%')
        
        # Evaluation
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs, return_rotation_labels=False)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            print(f'Finetune Epoch [{epoch+1}] Test Loss: {test_loss/len(test_loader):.4f} '
                  f'Test Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()

def standard_training(model, train_loader, test_loader, device, args):
    """기존 supervised learning"""
    print("=== Standard Supervised Training ===")
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                         momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
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
            
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Epoch: [{epoch+1}/{args.num_epochs}] '
                      f'Step: [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {train_loss/(batch_idx+1):.4f} '
                      f'Acc: {100.*correct/total:.2f}%')
        
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

def main(args):
    # num_classes 설정
    if args.dataset.lower() == "cifar100":
        args.num_classes = 100
    elif args.dataset.lower() == "cifar10":
        args.num_classes = 10
    else:
        args.num_classes = 10
    
    # 모델 로드 (args에 따라 모델별 파라미터 전달)
    model = load_model_with_args(args)
    
    # 데이터셋 로드 (SSL mode 고려)
    train_dataset = datasets.load_dataset(args.dataset, train=True, ssl_mode=args.ssl_mode)
    test_dataset = datasets.load_dataset(args.dataset, train=False, ssl_mode=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # SSL mode vs Standard mode
    if args.ssl_mode and args.model.lower() == "rotnet":
        # SSL: pretrain + finetune
        ssl_pretrain(model, train_loader, device, args)
        ssl_finetune(model, train_loader, test_loader, device, args)
    else:
        # Standard supervised training
        standard_training(model, train_loader, test_loader, device, args)
    
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Basic arguments
    parser.add_argument("--model", type=str, default="resnet34", 
                       help="Model: resnet34, densenet, fractalnet, preactresnet, rotnet")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                       help="Dataset: cifar10 or cifar100")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_step", type=int, default=30)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=1)
    
    # 모델별 특화 arguments 추가
    get_model_specific_args(parser)
    
    args = parser.parse_args()
    main(args)