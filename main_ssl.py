import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import models
import datasets
import frameworks

def ssl_training(framework, train_loader, device, args):
    """SSL training loop"""
    optimizer = optim.SGD(framework.parameters(), lr=args.lr, 
                         momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    framework.train()
    
    for epoch in range(args.ssl_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = framework.move_batch_to_device(batch, device)
            
            optimizer.zero_grad()
            loss = framework(batch)  # Framework handles batch processing
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Epoch [{epoch+1}/{args.ssl_epochs}] '
                      f'Step [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
                      
        scheduler.step()
        print(f'Epoch [{epoch+1}/{args.ssl_epochs}] Avg Loss: {epoch_loss/len(train_loader):.4f}')

def main(args):
    # Setup
    if args.dataset.lower() == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
        
    # Load encoder model
    encoder = models.load_model(args.model, num_classes=num_classes)
    
    # Initialize framework
    if args.framework == "rotnet":
        framework = frameworks.RotNet(encoder)
    else:
        raise ValueError(f"Framework {args.framework} not implemented")
    
    # Load datasets  
    train_dataset = datasets.load_dataset(args.dataset, train=True, ssl_mode=True)
    test_dataset = datasets.load_dataset(args.dataset, train=False, ssl_mode=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    framework.to(device)
    
    print("="*50)
    print(f"Framework: {args.framework}")
    print(f"Encoder: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"SSL Epochs: {args.ssl_epochs}")
    print("="*50)
    
    # SSL Training
    ssl_training(framework, train_loader, device, args)
    
    # k-NN Evaluation (using common functionality)
    print("Collecting training features...")
    train_features, train_labels = framework.collect_features(train_loader, device)
    
    print("Collecting test features...")
    test_features, test_labels = framework.collect_features(test_loader, device)
    
    print("Running k-NN evaluation...")
    accuracy = framework.knn_evaluation(train_features, train_labels, 
                                       test_features, test_labels, k=5)
    
    print(f"k-NN (k=5) Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--framework", type=str, default="rotnet",
                       choices=["rotnet"], help="Learning framework")
    parser.add_argument("--model", type=str, default="resnet34", 
                       help="Encoder model")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       help="Dataset: cifar10 or cifar100")
    parser.add_argument("--ssl_epochs", type=int, default=100,
                       help="SSL training epochs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_step", type=int, default=30)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--print_freq", type=int, default=50)
    
    args = parser.parse_args()
    main(args)