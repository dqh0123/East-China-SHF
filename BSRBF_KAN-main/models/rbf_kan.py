"""RBFKAN: 仅用RBF(全局)基函数, 无B-spline, 大幅减少参数量, 适配小样本"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RBFKANLayer(nn.Module):
    """
    简化的 KAN 层:
    - SiLU 基函数 (全局)
    - 仅 RBF 基函数 (全局, 无 B-spline)
    - 参数量 = input_dim×output_dim (base) + output_dim×(input_dim×num_rbf) (rbf)
    
    vs BSRBF_KANLayer:
    - 无 B-spline 展开 → 省去 (grid_size+spline_order) 的倍数扩增
    - 参数量降低约 (grid+spline)/num_rbf 倍
    """
    def __init__(self, input_dim, output_dim, num_rbf=5, grid_range=[-1.5, 1.5]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_rbf = num_rbf
        
        # LayerNorm
        self.ln = nn.LayerNorm(input_dim)
        
        # Base: SiLU + Linear
        self.base_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        
        # RBF 基函数 (全局 Gaussian)
        grid = torch.linspace(grid_range[0], grid_range[1], num_rbf)
        self.register_buffer('grid', grid)
        self.denom = (grid_range[1] - grid_range[0]) / (num_rbf - 1) if num_rbf > 1 else 1.0
        
        # RBF 权重
        rbf_in_dim = input_dim * num_rbf
        self.rbf_weight = nn.Parameter(torch.empty(output_dim, rbf_in_dim))
        nn.init.kaiming_uniform_(self.rbf_weight, a=math.sqrt(5))
    
    def forward(self, x):
        # LayerNorm
        x = self.ln(x)
        
        # Base: SiLU
        base_out = F.linear(F.silu(x), self.base_weight)
        
        # RBF: exp(-((x - center) / sigma)^2)
        # x: (batch, in_dim) → (batch, in_dim, 1)
        # grid: (num_rbf,) → (1, 1, num_rbf)
        rbf = torch.exp(-((x.unsqueeze(-1) - self.grid) / self.denom) ** 2)
        rbf = rbf.reshape(x.size(0), -1)  # (batch, in_dim * num_rbf)
        rbf_out = F.linear(rbf, self.rbf_weight)
        
        return base_out + rbf_out


class RBFKAN(nn.Module):
    """
    RBFKAN 网络, 与 BSRBF_KAN 使用相同的 layers_hidden 参数格式.
    例: RBFKAN([8, 32, 32, 12], num_rbf=5)
    """
    def __init__(self, layers_hidden, num_rbf=5):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(RBFKANLayer(in_dim, out_dim, num_rbf=num_rbf))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
