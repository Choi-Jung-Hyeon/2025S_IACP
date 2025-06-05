import argparse
import os

import models
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def main(args):
    # 선택한 모델에 따라 num_classes 설정
    if args.dataset.lower() == "cifar10" :
        num_classes = 10
    else :
        num_classes = 100
    model = models.load_model(args.model, num_classes=num_classes)
    print(f"Using {args.model} model")

    # dataset 호출
    dataset = datasets.load_dataset(args.dataset, train=True)
    print(f"Using {args.dataset} dataset")
    print(f"Using {args.batch} batch size")
    
    # DataLoader 생성: (여기서는 PyTorch의 DataLoader를 사용)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2)
    
    # 옵티마이저 정의 (예: SGD)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 간단한 training loop (에포크 수만큼 반복)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model.to(device)
    
    for epoch in range(args.epoch):
        model.train()

        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epoch}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

    print("Training finished.")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="resnet34", help="Choose the model: resnet34, densenet, fractalnet")
    argparser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use: cifar10 or cifar100")
    argparser.add_argument("--epoch", type=int, default=100, help="Number of training epochs")
    argparser.add_argument("--batch", type=int, default=64, help="Batch size")
    args = argparser.parse_args()
    main(args)
