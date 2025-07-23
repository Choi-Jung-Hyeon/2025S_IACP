import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import models
import datasets
import frameworks

def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = datasets.get_num_classes(args.dataset)
        
    # Load encoder model
    if args.model == "rotnet":
        encoder = models.rotnet(num_classes=4, num_blocks=args.num_blocks)
    else:
        encoder = models.load_model(args.model, num_classes=num_classes)
    
    # Initialize framework
    if args.framework == "rotnet":
        framework = frameworks.Rotnet(encoder)
    else:
        raise ValueError(f"Framework {args.framework} not implemented")

    # Load datasets
    train_dataset = datasets.load_dataset(args.dataset, train=True, ssl_mode=True)
    test_dataset = datasets.load_dataset(args.dataset, train=False, ssl_mode=False)
    train_labeled = datasets.load_dataset(args.dataset, train=True, ssl_mode=False)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)
    train_labeled_loader = DataLoader(train_labeled, batch_size=args.batch_size,
                                          shuffle=False, num_workers=4, pin_memory=True)
    
    framework.to(device)
    
    print("=" * 70)
    print("SELF-SUPERVISED LEARNING")
    print("=" * 70)
    print(f"Framework: {args.framework}, Encoder: {args.model}, Dataset: {args.dataset}")
    print(f"Optimizer: {args.optimizer.upper()}, Epochs: {args.ssl_epochs}, Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}, LR Milestones: {args.lr_milestones}, LR Gamma: {args.lr_gamma}")
    if args.model == "rotnet":
        print(f"NIN Blocks: {args.num_blocks}")
    print("=" * 70)
    
    # Optimizer & Scheduler
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(framework.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(framework.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(framework.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    
    # 마지막 1-NN 정확도 저장을 위한 변수
    last_1nn_accuracy = 0.0

    # 학습 및 평가 루프
    for epoch in range(args.ssl_epochs):
        framework.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = framework.move_batch_to_device(batch, device)
            
            optimizer.zero_grad()
            loss = framework(batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Epoch [{epoch+1}/{args.ssl_epochs}] '
                      f'Step [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
                      
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch+1}/{args.ssl_epochs}] Avg Loss: {avg_loss:.4f} LR: {current_lr:.6f}')

        # 10 에포크마다 또는 마지막 에포크에서 1-NN 평가 수행
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.ssl_epochs:
            framework.eval()

            train_features, train_labels = framework.collect_features(train_labeled_loader, device)
            test_features, test_labels = framework.collect_features(test_loader, device)
            accuracy = framework.knn_evaluation(train_features, train_labels, 
                                                test_features, test_labels)
            
            accuracy_percent = accuracy * 100
            print(f"Epoch [{epoch+1}] 1-NN Test Accuracy: {accuracy_percent:.2f}%")
            
            # 마지막 정확도 업데이트
            last_1nn_accuracy = accuracy_percent
            
            framework.train() # 다시 학습 모드로 전환

    print("=" * 70)
    print("SELF-SUPERVISED LEARNING FINISHED")
    print(f"Final 1-NN Test Accuracy: {last_1nn_accuracy:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self-Supervised Learning')
    
    # Framework & Model
    parser.add_argument("--framework", type=str, default="rotnet", choices=["rotnet"], help="SSL framework")
    parser.add_argument("--model", type=str, default="rotnet", help="Encoder: rotnet, resnet34, etc.")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset: cifar10 or cifar100")
    
    # Optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"], help="Optimizer type")

    # Training Hyperparameters
    parser.add_argument("--ssl_epochs", type=int, default=100, help="SSL training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--lr_milestones", nargs='+', type=int, default=[30, 70, 80], help="Epochs to decay LR")
    parser.add_argument("--lr_gamma", type=float, default=0.2, help="LR decay factor (e.g., 0.2 for RotNet)")
    
    # Logging
    parser.add_argument("--print_freq", type=int, default=50, help="Print frequency")
    
    # RotNet specific
    parser.add_argument("--num_blocks", type=int, default=4, choices=[3, 4, 5], help="NIN blocks for RotNet model")
    
    args = parser.parse_args()
    main(args)
