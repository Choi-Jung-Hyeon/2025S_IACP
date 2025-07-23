import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import models
import datasets
import frameworks

def get_optimizer(model, args):
    return optim.SGD(model.parameters(), lr=args.lr, 
                     momentum=0.9, weight_decay=args.weight_decay)

def get_scheduler(optimizer, args, is_ssl=False):
    if is_ssl:
        return MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
    
    if args.model == "fractalnet":
        return StepLR(optimizer, step_size=200, gamma=0.1)
    elif args.model == "densenet":
        return StepLR(optimizer, step_size=150, gamma=0.1)
    else:
        return StepLR(optimizer, step_size=100, gamma=0.1)

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        if hasattr(model, 'move_batch_to_device'):
            batch = model.move_batch_to_device(batch, device)
        else:
            batch = batch.to(device) if torch.is_tensor(batch) else \
                     tuple(b.to(device) for b in batch)
        
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.encoder(inputs)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total

def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = datasets.get_num_classes(args.dataset)
    is_ssl = args.framework is not None
    
    # Model
    encoder = models.load_model(args.model, num_classes=num_classes, 
                               num_blocks=getattr(args, 'num_blocks', 4))
    
    if is_ssl:
        model = getattr(frameworks, args.framework.capitalize())(encoder)
    else:
        model = frameworks.SupervisedLearning(encoder)
    
    model.to(device)
    
    # Data
    train_dataset = datasets.load_dataset(args.dataset, train=True, ssl_mode=is_ssl)
    test_dataset = datasets.load_dataset(args.dataset, train=False)
    
    batch_size = 128 if is_ssl else (100 if args.model == "fractalnet" else 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Adjust lr and epochs for FractalNet
    if args.model == "fractalnet" and not is_ssl:
        args.lr = 0.02
        args.num_epochs = 400
    elif is_ssl:
        args.num_epochs = 100
    
    # Training setup
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args, is_ssl)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    for epoch in range(1, args.num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        if not is_ssl and epoch % 10 == 0:
            test_loss, acc = evaluate(model, test_loader, criterion, device)
            print(f'Epoch {epoch}: Loss={loss:.4f} Test Loss={test_loss:.4f} Acc={acc:.2f}%')
        
        scheduler.step()
    
    # Final evaluation
    if is_ssl:
        train_labeled = datasets.load_dataset(args.dataset, train=True)
        train_labeled_loader = DataLoader(train_labeled, batch_size=batch_size,
                                        shuffle=False, num_workers=4)
        
        train_feats, train_labels = model.collect_features(train_labeled_loader, device)
        test_feats, test_labels = model.collect_features(test_loader, device)
        accuracy = model.knn_evaluation(train_feats, train_labels, test_feats, test_labels, k=1)
        print(f'\n1-NN Accuracy: {accuracy*100:.2f}%')
    else:
        test_loss, acc = evaluate(model, test_loader, criterion, device)
        print(f'\nFinal Test Accuracy: {acc:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="preactresnet", 
                       help="Model: resnet34, densenet, fractalnet, preactresnet, rotnet")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       help="Dataset: cifar10 or cifar100")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=1)
    args = parser.parse_args()
    main(args)