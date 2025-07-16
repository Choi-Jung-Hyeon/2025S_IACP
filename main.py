import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import models
import datasets

def get_training_config(model_name):
    """Get training config based on paper specifications"""
    configs = {
        "fractalnet": {
            "lr": 0.02,  # FractalNet paper: 0.02
            "batch_size": 100,  # FractalNet paper: 100
            "num_epochs": 400,  # FractalNet paper: 400
            "lr_step": 200,  # Drop by 10 when remaining epochs halve
            "lr_gamma": 0.1
        },
        "densenet": {
            "lr": 0.1,
            "batch_size": 64,
            "num_epochs": 300,
            "lr_step": 150,
            "lr_gamma": 0.1
        },
        "default": {
            "lr": 0.1,
            "batch_size": 64,
            "num_epochs": 200,
            "lr_step": 100,
            "lr_gamma": 0.1
        }
    }
    return configs.get(model_name, configs["default"])

def main(args):
    # Get number of classes
    num_classes = datasets.get_num_classes(args.dataset)
    
    # Load model
    model = models.load_model(args.model, num_classes=num_classes)
    
    # Load datasets
    train_dataset = datasets.load_dataset(args.dataset, train=True, ssl_mode=False)
    test_dataset = datasets.load_dataset(args.dataset, train=False, ssl_mode=False)
    
    # Get paper-specific config if not overridden
    config = get_training_config(args.model)
    final_lr = args.lr if args.lr != 0.1 else config["lr"]
    final_batch_size = args.batch_size if args.batch_size != 64 else config["batch_size"]
    final_epochs = args.num_epochs if args.num_epochs != 200 else config["num_epochs"]
    final_lr_step = args.lr_step if args.lr_step != 100 else config["lr_step"]
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=final_batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=final_batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Optimizer, criterion, scheduler
    optimizer = optim.SGD(model.parameters(), lr=final_lr, 
                         momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=final_lr_step, gamma=args.lr_gamma)
    
    print("=" * 70)
    print("SUPERVISED LEARNING")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Num Classes: {num_classes}")
    print(f"Num Epochs: {final_epochs}")
    print(f"Batch Size: {final_batch_size}")
    print(f"Learning Rate: {final_lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"LR Scheduler Step: {final_lr_step}")
    print(f"LR Scheduler Gamma: {args.lr_gamma}")
    if args.model == "fractalnet":
        print("🔥 Using FractalNet paper settings (LR=0.02, BS=100, Epochs=400)")
    print("=" * 70)
    
    # Training loop
    for epoch in range(final_epochs):
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
                print(f'Epoch: [{epoch+1}/{final_epochs}] '
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
    parser = argparse.ArgumentParser(description='Supervised Learning')
    parser.add_argument("--model", type=str, default="resnet34", 
                       help="Model: resnet34, densenet, fractalnet, preactresnet")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                       help="Dataset: cifar10 or cifar100")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_step", type=int, default=100)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=1)
    
    args = parser.parse_args()
    main(args)