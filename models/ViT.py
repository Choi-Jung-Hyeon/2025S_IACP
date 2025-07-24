import torch
import torch.nn as nn
import math

class Patch_Embedding(nn.Module):
    def __init__(self, in_channels, height, width, patch_size, d_model):
        super().__init__()

        self.h = height
        self.w = width
        self.p = patch_size
        self.c = in_channels
        self.n = (height * width) // (patch_size ** 2)
        
        patch_dim = in_channels * (patch_size ** 2)
        self.projection = nn.Linear(patch_dim, d_model)
        
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_encoding = nn.Parameter(torch.randn(1, self.n + 1, d_model))

    def patch(self, x):
        p = self.p
        out = x.reshape(-1, self.c, self.h // p, p, self.w // p, p)
        out = out.permute(0, 2, 4, 1, 3, 5) # b, h//p, w//p, c, p, p
        out = out.reshape(-1, self.n, self.c * p * p) # b, n, c*p*p
        return out
    
    def forward(self, x):
        out = self.patch(x)
        out = self.projection(out)
        
        cls_tokens = self.class_token.expand(x.shape[0], -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)
        
        out += self.positional_encoding
        return out

class Hybrid_Embedding(nn.Module):
    def __init__(self, in_channels, height, width, patch_size, d_model):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.n = (height * width) // (patch_size ** 2)
        
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_encoding = nn.Parameter(torch.randn(1, self.n + 1, d_model))
        
    def forward(self, x):
        out = self.conv(x).flatten(2).transpose(1, 2)
        
        cls_tokens = self.class_token.expand(x.shape[0], -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)

        out += self.positional_encoding
        return out

class MSA(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        
        self.h = num_head
        self.d_model = d_model
        self.d_k = d_model // num_head
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.MSA_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        b, n, d = x.shape
        
        # (B, N, D) -> (B, N, H, D/H) -> (B, H, N, D/H)
        q = self.q_linear(x).view(b, n, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(b, n, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(b, n, self.h, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_softmax = torch.softmax(attn_scores, dim=-1)
        
        out = torch.matmul(attn_softmax, v) # (B, H, N, D/H)
        
        # (B, H, N, D/H) -> (B, N, H, D/H) -> (B, N, D)
        out = out.transpose(1, 2).contiguous().view(b, n, d)

        out = self.MSA_linear(out)
        return out

class Encoder(nn.Module):
    def __init__(self, d_model, num_head, MLP_size):
        super().__init__()
        self.MSA = MSA(d_model, num_head)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, MLP_size),
            nn.GELU(),
            nn.Linear(MLP_size, d_model),
        ) 
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Pre-LN (Layer Normalization -> Sublayer -> Residual)
        x = x + self.MSA(self.LN1(x))
        x = x + self.MLP(self.LN2(x))
        return x

class vit(nn.Module):
    def __init__(self, num_classes, in_channels=3, height=32, width=32, patch_size=4, 
                 d_model=192, num_layers=12, num_head=12, MLP_size=384, hybrid=False):
        super().__init__()
        
        if hybrid:
            self.Embedding = Hybrid_Embedding(in_channels, height, width, patch_size, d_model)
        else:
            self.Embedding = Patch_Embedding(in_channels, height, width, patch_size, d_model)
        
        self.Encoder_layers = nn.ModuleList([
            Encoder(d_model, num_head, MLP_size) for _ in range(num_layers)
        ])
        
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        x = self.Embedding(x)
        
        for layer in self.Encoder_layers:
            x = layer(x)

        x = self.MLP_head(x[:, 0])
        
        return x
    
    def _extract_features(self, x):
        x = self.Embedding(x)
        
        for layer in self.Encoder_layers:
            x = layer(x)

        return x[:, 0]
