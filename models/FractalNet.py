import torch
import torch.nn as nn

# BasicConv: Conv + BN + ReLU
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# FractalBlock:
# Base: f1(z)=conv(z)
# Recursion
# join = element-wise mean
class FractalBlock(nn.Module):
    def __init__(self, channels, depth):
        super(FractalBlock, self).__init__()
        self.depth = depth
        if depth == 1:
            self.block = BasicConv(channels, channels)
        else:
            self.branch = FractalBlock(channels, depth - 1)
            self.conv = BasicConv(channels, channels)
            
    def forward(self, x):
        if self.depth == 1:
            return self.block(x)
        out1 = self.branch(self.branch(x))
        out2 = self.conv(x)
        return (out1 + out2) / 2  # join: mean

# FractalNet:
# initial conv -> fractal block -> global pool -> fc
class FractalNet(nn.Module):
    def __init__(self, num_classes=100, fractal_depth=3):
        super(FractalNet, self).__init__()
        self.initial = BasicConv(3, 64, kernel_size=3, stride=1, padding=1)
        self.fractal = FractalBlock(64, fractal_depth)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.initial(x)   # initial
        x = self.fractal(x)   # fractal branch
        x = self.pool(x)      # global pool
        x = torch.flatten(x, 1)
        return self.fc(x)     # fc

# load_model: dispatcher for main.py
def load_model(model_name):
    if model_name.lower() == "fractalnet":
        return FractalNet(num_classes=100, fractal_depth=3)
    raise ValueError("Unsupported model: {}".format(model_name))

if __name__ == "__main__":
    # Test: dummy input (CIFAR size)
    model = load_model("fractalnet")
    print(model)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
