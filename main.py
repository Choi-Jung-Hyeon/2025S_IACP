import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import matplotlib.pyplot as plt
import os

import models
import datasets
import frameworks

def plot_ssl_accuracy_graph(eval_epochs, test_accs, args):
    plt.figure(figsize=(10, 6))
    plt.plot(eval_epochs, test_accs, 'o-', label='1-NN Test Accuracy')

    title = f'1-NN Accuracy over Epochs\nModel: {args.model}, Framework: {args.framework}, Dataset: {args.dataset}'
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max(100, max(test_accs) * 1.1 if test_accs else 100))

    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)
    file_name = f'{args.model}_{args.dataset}_{args.framework}_1nn_accuracy.png'
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    
    print(f"\n1-NN Accuracy plot saved to: {save_path}")

def plot_supervised_accuracy_graph(epochs, train_accs, test_accs, args):
    plt.figure(figsize=(10, 6))
    epoch_range = range(1, epochs + 1)
    
    plt.plot(epoch_range, train_accs, '-', label='Training Accuracy')
    plt.plot(epoch_range, test_accs, '-', label='Test Accuracy')
    
    title = f'Supervised Accuracy\nModel: {args.model}, Dataset: {args.dataset}, Optimizer: {args.optimizer.upper()}'
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    
    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)
    file_name = f'{args.model}_{args.dataset}_{args.optimizer}_accuracy.png'
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    
    print(f"\nAccuracy plot saved to: {save_path}")

def run_ssl(args):
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
    elif args.framework == "simclr":
        framework = frameworks.SimCLR(encoder)
    elif args.framework == "simsiam":
        framework = frameworks.SimSiam(encoder)
    else:
        raise ValueError(f"Framework {args.framework} is not a valid SSL framework")

    # Load datasets for SSL
    train_dataset = datasets.load_dataset(args.dataset, train=True, ssl_framework=args.framework)
    test_dataset = datasets.load_dataset(args.dataset, train=False, ssl_framework=None)
    train_labeled = datasets.load_dataset(args.dataset, train=True, ssl_framework=None)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_labeled_loader = DataLoader(train_labeled, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    framework.to(device)
    
    print("=" * 70)
    print("SELF-SUPERVISED LEARNING")
    print("=" * 70)
    print(f"Framework: {args.framework}, Encoder: {args.model}, Dataset: {args.dataset}")
    print(f"Optimizer: {args.optimizer.upper()}, Epochs: {args.num_epochs}, Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}, LR Milestones: {args.lr_milestones}, LR Gamma (SSL): {args.ssl_lr_gamma}")
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

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.ssl_lr_gamma)
    
    eval_epochs = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(args.num_epochs):
        framework.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = framework.move_batch_to_device(batch, device)
            
            optimizer.zero_grad()
            loss = framework(batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Epoch: [{epoch+1}/{args.num_epochs}] | Step: [{batch_idx+1}/{len(train_loader)}] | Loss: {train_loss/(batch_idx+1):.4f}')
        
        # 1-NN Evaluation every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n--- Evaluating at epoch {epoch+1} ---")
            
            # Collect features
            train_features, train_labels = framework.collect_features(train_labeled_loader, device)
            test_features, test_labels = framework.collect_features(test_loader, device)
            
            # 1-NN evaluation
            accuracy = framework.knn_evaluation(train_features, train_labels, test_features, test_labels)
            
            eval_epochs.append(epoch+1)
            test_accuracies.append(accuracy.item() * 100)
            
            print(f'Epoch [{epoch+1}] 1-NN Test Accuracy: {accuracy.item()*100:.2f}%')
            
            # Compute test loss
            framework.eval()
            test_loss = 0.0
            batch_count = 0
            with torch.no_grad():
                for batch in test_loader:
                    if batch_count >= 10:  # First 10 batches for test loss
                        break
                    batch = framework.move_batch_to_device(batch, device)
                    loss = framework(batch)
                    test_loss += loss.item()
                    batch_count += 1
            print(f'Epoch [{epoch+1}] Test Loss (first 10 batches): {test_loss/10:.4f}')
        
        scheduler.step()
    
    print("\nSelf-Supervised Learning Finished.")
    if eval_epochs:
        plot_ssl_accuracy_graph(eval_epochs, test_accuracies, args)

def run_supervised(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = datasets.get_num_classes(args.dataset)
    model = models.load_model(args.model, num_classes=num_classes)
    model.to(device)
    
    train_dataset = datasets.load_dataset(args.dataset, train=True, ssl_framework=None)
    test_dataset = datasets.load_dataset(args.dataset, train=False, ssl_framework=None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Optimizer
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.sup_lr_gamma)
    
    print("=" * 70)
    print("SUPERVISED LEARNING")
    print("=" * 70)
    print(f"Model: {args.model}, Dataset: {args.dataset}, Optimizer: {args.optimizer.upper()}")
    print(f"Epochs: {args.num_epochs}, Batch Size: {args.batch_size}, Learning Rate: {args.lr}")
    print(f"LR Step: {args.lr_step}, LR Gamma (Supervised): {args.sup_lr_gamma}")
    print("=" * 70)

    train_accuracies, test_accuracies = [], []
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
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
                print(f'Epoch: [{epoch+1}/{args.num_epochs}] | Step: [{batch_idx+1}/{len(train_loader)}] | Acc: {100.*correct_train/total_train:.2f}%')

        epoch_train_acc = 100. * correct_train / total_train
        train_accuracies.append(epoch_train_acc)
        
        # Evaluation
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
        
        epoch_test_acc = 100. * correct_test / total_test
        test_accuracies.append(epoch_test_acc)
        
        print(f'Epoch [{epoch+1}] Test Acc: {epoch_test_acc:.2f}%')
        
        # Print test loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}] Test Loss: {test_loss/len(test_loader):.4f}')
        
        scheduler.step()

    print("\nSupervised Learning Finished.")
    plot_supervised_accuracy_graph(args.num_epochs, train_accuracies, test_accuracies, args)

def main(args):
    if args.framework == 'supervised':
        run_supervised(args)
    else:
        run_ssl(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SSL and Supervised Learning Framework')
    
    # Common arguments
    common_parser = parser.add_argument_group('Common Arguments')
    common_parser.add_argument("--framework", type=str, default="supervised", 
                              choices=["rotnet", "simclr", "simsiam", "supervised"], help="Framework")
    common_parser.add_argument("--model", type=str, default="resnet18", help="Encoder model")
    common_parser.add_argument("--dataset", type=str, default="cifar10", 
                              choices=["cifar10", "cifar100"], help="Dataset")
    common_parser.add_argument("--optimizer", type=str, default="sgd", 
                              choices=["sgd", "adam", "adamw"], help="Optimizer")
    common_parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    common_parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    common_parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    common_parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    common_parser.add_argument("--print_freq", type=int, default=50, help="Print frequency")

    # Supervised Learning arguments
    sup_parser = parser.add_argument_group('Supervised Learning Arguments')
    sup_parser.add_argument("--lr_step", type=int, default=30, help="Step size for StepLR scheduler")
    sup_parser.add_argument("--sup_lr_gamma", type=float, default=0.1, help="Gamma for StepLR scheduler")

    # Self-Supervised Learning arguments
    ssl_parser = parser.add_argument_group('Self-Supervised Learning Arguments')
    ssl_parser.add_argument("--lr_milestones", nargs='+', type=int, default=[60, 80], 
                           help="Epochs to decay LR for MultiStepLR")
    ssl_parser.add_argument("--ssl_lr_gamma", type=float, default=0.2, help="Gamma for MultiStepLR scheduler")
    ssl_parser.add_argument("--num_blocks", type=int, default=4, choices=[3, 4, 5], 
                           help="NIN blocks for RotNet model")
    
    args = parser.parse_args()
    main(args)
