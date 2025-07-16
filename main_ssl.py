import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import models
import datasets
import frameworks

def ssl_training(framework, train_loader, device, args):
    """SSL training loop (논문 기반)"""
    # RotNet paper: SGD, momentum=0.9, weight_decay=5e-4, lr=0.1
    optimizer = optim.SGD(framework.parameters(), lr=args.lr, 
                         momentum=0.9, weight_decay=args.weight_decay)
    
    # RotNet paper: Drop lr by factor of 5 after epochs 30, 60, 80
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
    
    framework.train()
    
    for epoch in range(args.ssl_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Framework에서 batch device 이동 처리 (7월 10일 Comment(1))
            batch = framework.move_batch_to_device(batch, device)
            
            optimizer.zero_grad()
            loss = framework(batch)  # Framework가 알아서 처리
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

def main(args):
    # Setup
    num_classes = datasets.get_num_classes(args.dataset)
        
    # Load encoder model (7월 10일 Comment(1))
    if args.model == "rotnet":
        encoder = models.rotnet(num_classes=4, num_blocks=args.num_blocks)
    else:
        encoder = models.load_model(args.model, num_classes=num_classes)
    
    # Initialize framework
    if args.framework == "rotnet":
        framework = frameworks.RotNet(encoder)
    else:
        raise ValueError(f"Framework {args.framework} not implemented")
    
    # Load datasets
    train_dataset = datasets.load_dataset(args.dataset, train=True, ssl_mode=True)
    test_dataset = datasets.load_dataset(args.dataset, train=False, ssl_mode=False)
    
    # Separate labeled dataset for k-NN evaluation
    train_labeled = datasets.load_dataset(args.dataset, train=True, ssl_mode=False)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    train_labeled_loader = DataLoader(train_labeled, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    framework.to(device)
    
    print("=" * 60)
    print("SELF-SUPERVISED LEARNING (RotNet)")
    print("=" * 60)
    print(f"Framework: {args.framework}")
    print(f"Encoder: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"SSL Epochs: {args.ssl_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    if args.model == "rotnet":
        print(f"NIN Blocks: {args.num_blocks}")
    print("=" * 60)
    
    # SSL Training
    ssl_training(framework, train_loader, device, args)
    
    # k-NN Evaluation (7월 10일 Comment(1))
    print("\n" + "="*60)
    print("k-NN EVALUATION")
    print("="*60)
    
    print("Collecting training features...")
    train_features, train_labels = framework.collect_features(train_labeled_loader, device)
    
    if train_features is None:
        print("❌ No labeled training data available for k-NN evaluation")
        return
    
    print("Collecting test features...")
    test_features, test_labels = framework.collect_features(test_loader, device)
    
    print("Running k-NN evaluation...")
    accuracy = framework.knn_evaluation(train_features, train_labels, 
                                       test_features, test_labels, k=5)
    
    print(f"\n✅ Final k-NN (k=5) Best Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self-Supervised Learning')
    
    # Framework
    parser.add_argument("--framework", type=str, default="rotnet",
                       choices=["rotnet"], help="SSL framework")
    parser.add_argument("--model", type=str, default="rotnet", 
                       help="Encoder: rotnet, resnet34, densenet, fractalnet, preactresnet")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       help="Dataset: cifar10 or cifar100")
    
    # Training
    parser.add_argument("--ssl_epochs", type=int, default=100,
                       help="SSL training epochs (논문: 100)")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size (논문: 128)")
    parser.add_argument("--lr", type=float, default=0.1,
                       help="Learning rate (논문: 0.1)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                       help="Weight decay (논문: 5e-4)")
    parser.add_argument("--print_freq", type=int, default=50)
    
    # RotNet specific
    parser.add_argument("--num_blocks", type=int, default=4,
                       choices=[3, 4, 5], help="NIN blocks (논문: 3,4,5)")
    
    args = parser.parse_args()
    main(args)