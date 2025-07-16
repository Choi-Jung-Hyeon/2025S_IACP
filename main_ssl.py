import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.neighbors import NearestNeighbors

import models
import datasets
import frameworks

def move_batch_to_device(batch, device):
    """Move batch (tuple or tensor) to device"""
    if isinstance(batch, tuple):
        return tuple(item.to(device) for item in batch)
    return batch.to(device)

def ssl_training(framework, train_loader, device, args):
    """Self-supervised learning training"""
    optimizer = optim.SGD(framework.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    framework.train()
    
    for epoch in range(args.ssl_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            
            optimizer.zero_grad()
            loss = framework(batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Epoch [{epoch+1}/{args.ssl_epochs}] '
                      f'Step [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
                      
        scheduler.step()
        print(f'Epoch [{epoch+1}/{args.ssl_epochs}] Avg Loss: {epoch_loss/len(train_loader):.4f}')

def collect_features(framework, data_loader, device):
    """Collect features and labels for k-NN"""
    framework.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            x, y = batch
            
            features = framework.extract_features(x)
            features = features.view(features.size(0), -1)  # Flatten
            
            all_features.append(features.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
    return np.vstack(all_features), np.hstack(all_labels)

def knn_evaluation(train_features, train_labels, test_features, test_labels, k=5):
    """k-NN classifier evaluation"""
    # Fit k-NN classifier
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(train_features)
    
    # Find k nearest neighbors for test features
    distances, indices = knn.kneighbors(test_features)
    
    # Majority vote
    correct = 0
    for i, neighbors in enumerate(indices):
        neighbor_labels = train_labels[neighbors]
        predicted_label = np.bincount(neighbor_labels).argmax()
        if predicted_label == test_labels[i]:
            correct += 1
            
    accuracy = correct / len(test_labels)
    return accuracy

def main(args):
    # Load encoder model
    if args.dataset.lower() == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
        
    encoder = models.load_model(args.model, num_classes=num_classes)
    
    # Initialize framework
    if args.framework == "rotnet":
        framework = frameworks.RotNet(encoder)
    else:
        framework = frameworks.SupervisedLearning(encoder)
    
    # Load datasets
    train_dataset = datasets.load_dataset(args.dataset, train=True)
    test_dataset = datasets.load_dataset(args.dataset, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    framework.to(device)
    
    print("="*50)
    print(f"Framework: {args.framework}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"SSL Epochs: {args.ssl_epochs}")
    print("="*50)
    
    # SSL Training
    ssl_training(framework, train_loader, device, args)
    
    # Feature collection for k-NN evaluation
    print("Collecting training features...")
    train_features, train_labels = collect_features(framework, train_loader, device)
    
    print("Collecting test features...")
    test_features, test_labels = collect_features(framework, test_loader, device)
    
    # k-NN evaluation
    print("Running k-NN evaluation...")
    accuracy = knn_evaluation(train_features, train_labels, test_features, test_labels, k=5)
    
    print(f"k-NN (k=5) Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--framework", type=str, default="rotnet", choices=["rotnet", "supervised"], help="Learning framework")
    parser.add_argument("--model", type=str, default="resnet34", help="Encoder model")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset: cifar10 or cifar100")
    parser.add_argument("--ssl_epochs", type=int, default=100, help="SSL training epochs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_step", type=int, default=30)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--print_freq", type=int, default=50)
    
    args = parser.parse_args()
    main(args)