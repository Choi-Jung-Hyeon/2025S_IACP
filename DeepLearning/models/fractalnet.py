import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, columns, stride=1):
        super().__init__()
        self.columns = columns
        self.stride = stride
        
        self.layers = nn.ModuleDict()
        
        # Base case: f^1
        self.layers['conv_1_1'] = ConvBlock(in_channels, out_channels, stride=stride)
        
        # Recursive case: f^c
        for c in range(2, columns + 1):
            for i in range(2**(c-1)):
                layer_name = f'conv_{c}_{i+1}'
                if i == 0 and stride == 2:
                    self.layers[layer_name] = ConvBlock(in_channels if c == 2 else out_channels, 
                                                      out_channels, stride=stride)
                else:
                    self.layers[layer_name] = ConvBlock(out_channels, out_channels)
    
    def _get_fractal_paths(self, x, column, drop_path_prob=0.0):
        if column == 1:
            return [self.layers['conv_1_1'](x)]
        
        paths = []
        
        # Deeper path
        prev_paths = self._get_fractal_paths(x, column - 1, drop_path_prob)
        for path_output in prev_paths:
            deeper_paths = self._get_fractal_paths(path_output, column - 1, drop_path_prob)
            paths.extend(deeper_paths)
        
        # Shallow path
        conv_layer = self.layers[f'conv_{column}_1']
        paths.append(conv_layer(x))
        
        return paths
    
    def forward(self, x, drop_path_prob=0.0, global_column=None):
        if global_column is not None:
            # Global drop-path
            paths = self._get_fractal_paths(x, global_column, drop_path_prob)
            if len(paths) == 1:
                return paths[0]
            return self._join_paths(paths, drop_path_prob)
        
        # All columns
        all_outputs = []
        for c in range(1, self.columns + 1):
            paths = self._get_fractal_paths(x, c, drop_path_prob)
            if len(paths) == 1:
                column_output = paths[0]
            else:
                column_output = self._join_paths(paths, drop_path_prob)
            all_outputs.append(column_output)
        
        return self._join_paths(all_outputs, drop_path_prob)
    
    def _join_paths(self, paths, drop_path_prob=0.0):
        if len(paths) == 1:
            return paths[0]
        
        if self.training and drop_path_prob > 0.0:
            # Local drop-path
            active_mask = torch.rand(len(paths)) > drop_path_prob
            if not active_mask.any():
                active_mask[torch.randint(0, len(paths), (1,))] = True
            
            active_paths = [paths[i] for i in range(len(paths)) if active_mask[i]]
            num_active = len(active_paths)
        else:
            active_paths = paths
            num_active = len(paths)
        
        # Element-wise mean
        if num_active == 1:
            return active_paths[0]
        else:
            result = active_paths[0]
            for path in active_paths[1:]:
                result = result + path
            return result / num_active

class SimplifiedFractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, columns, stride=1):
        super().__init__()
        self.columns = columns
        self.stride = stride
        
        self.column_paths = nn.ModuleList()
        
        for c in range(1, columns + 1):
            # Column c: 2^(c-1) layers
            num_layers = 2**(c-1)
            column = nn.ModuleList()
            
            for i in range(num_layers):
                if i == 0 and stride == 2:
                    layer = ConvBlock(in_channels, out_channels, stride=stride)
                else:
                    layer = ConvBlock(out_channels if i > 0 else in_channels, out_channels)
                column.append(layer)
            
            self.column_paths.append(column)
    
    def forward(self, x, drop_path_prob=0.0, global_column=None):
        if global_column is not None:
            return self._forward_column(x, global_column - 1)
        
        column_outputs = []
        for c in range(self.columns):
            output = self._forward_column(x, c)
            column_outputs.append(output)
        
        return self._join_with_drop_path(column_outputs, drop_path_prob)
    
    def _forward_column(self, x, column_idx):
        out = x
        for layer in self.column_paths[column_idx]:
            out = layer(out)
        return out
    
    def _join_with_drop_path(self, outputs, drop_path_prob=0.0):
        if len(outputs) == 1:
            return outputs[0]
        
        if self.training and drop_path_prob > 0.0:
            keep_mask = torch.rand(len(outputs), device=outputs[0].device) > drop_path_prob
            if not keep_mask.any():
                keep_mask[torch.randint(0, len(outputs), (1,))] = True
            
            active_outputs = [outputs[i] for i in range(len(outputs)) if keep_mask[i]]
        else:
            active_outputs = outputs
        
        if len(active_outputs) == 1:
            return active_outputs[0]
        
        result = torch.stack(active_outputs, dim=0).mean(dim=0)
        return result

class FractalNet(nn.Module):
    def __init__(self, num_classes=10, num_columns=4, channels=[64, 128, 256, 512], 
                 drop_path_prob=0.15, simplified=True):
        super().__init__()
        self.num_columns = num_columns
        self.drop_path_prob = drop_path_prob
        
        # Initial conv
        self.conv1 = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1)
        
        # Fractal blocks
        self.fractal_blocks = nn.ModuleList()
        in_channels = 64
        
        for i, out_channels in enumerate(channels):
            stride = 1 if i == 0 else 2
            
            if simplified:
                block = SimplifiedFractalBlock(in_channels, out_channels, num_columns, stride)
            else:
                block = FractalBlock(in_channels, out_channels, num_columns, stride)
            
            self.fractal_blocks.append(block)
            in_channels = out_channels
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(channels[-1], num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, global_drop_path=False):
        x = self.conv1(x)
        
        # 50% local + 50% global
        use_global = global_drop_path or (self.training and torch.rand(1).item() < 0.5)
        
        if use_global and self.training:
            selected_column = torch.randint(1, self.num_columns + 1, (1,)).item()
            for block in self.fractal_blocks:
                x = block(x, drop_path_prob=0.0, global_column=selected_column)
        else:
            for block in self.fractal_blocks:
                x = block(x, drop_path_prob=self.drop_path_prob)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Factory function
def fractalnet_cifar(num_classes=10, num_columns=4, simplified=True):
    return FractalNet(
        num_classes=num_classes,
        num_columns=num_columns,
        channels=[64, 128, 256, 512],
        drop_path_prob=0.15,
        simplified=simplified
    )

# Test
if __name__ == "__main__":
    print("Testing Simplified FractalNet...")
    model = fractalnet_cifar(num_classes=100, num_columns=4, simplified=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(2, 3, 32, 32)
    
    # Normal forward
    model.eval()
    with torch.no_grad():
        y = model(x)
        print(f"Output shape: {y.shape}")
    
    # Training with drop-path
    model.train()
    y_train = model(x)
    print(f"Training output shape: {y_train.shape}")
    
    # Global drop-path
    y_global = model(x, global_drop_path=True)
    print(f"Global drop-path output shape: {y_global.shape}")
    
    # Network structure
    print("\nNetwork structure:")
    for i, block in enumerate(model.fractal_blocks):
        depths = [len(column) for column in block.column_paths]
        print(f"Block {i+1} column depths: {depths}")