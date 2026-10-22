# =============================================================================
# Third-party attribution and license notice
# -----------------------------------------------------------------------------
# This file is an UNMODIFIED copy of "models/mlp.py" from the
# BSRBF_KAN project:
#     https://github.com/hoangthangta/BSRBF_KAN
#     Combine B-Splines (BS) and Radial Basis Functions (RBF) in
#     Kolmogorov-Arnold Networks (KANs)
#     Author: Hoang-Thang Ta
#
# Copyright (c) 2024-2026 Hoang-Thang Ta
#
# License: the upstream snapshot that this file was copied from is
# distributed under the MIT License (Copyright (c) 2024 Hoang Thang Ta).
# The project's current license, whose terms this repository also complies
# with, is the "ACADEMIC AND NON-COMMERCIAL LICENSE" (Copyright (c) 2026
# Hoang-Thang Ta): use, copying, modification and distribution are permitted
# solely for educational, academic-research and other non-commercial research
# purposes, and commercial use requires prior written permission from the
# copyright holder. The complete license text is kept in
# ../THIRD_PARTY_LICENSES/BSRBF_KAN-LICENSE.txt and must accompany any
# redistribution.
#
# No functional changes were made to this file. It is used by ASG-GKANWR as
# the MLP (multilayer perceptron) baseline model. It is not an official
# version developed or endorsed by Hoang-Thang Ta.
#
# If you use this code in academic work, please cite the associated paper:
#   Ta, H. T. (2025). BSRBF-KAN: A combination of B-splines and Radial Basis
#   Functions in Kolmogorov-Arnold Networks. In W. Buntine et al. (Eds.),
#   Information and Communication Technology (pp. 3-15). Springer Nature.
#   https://doi.org/10.1007/978-981-96-4288-5_1
#
# Based on software developed by Hoang-Thang Ta.
# =============================================================================

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_activation = torch.nn.SiLU,

    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.output_dim = output_dim
        self.base_activation = base_activation()
        self.input_dim = input_dim
        
        self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        #self.drop = nn.Dropout(p=0.01) # dropout
 
    def forward(self, x):
    
        # layer normalization
        x = self.layernorm(x)
        
        #x = self.drop(x)
        base_output = F.linear(self.base_activation(x), self.base_weight)

        return base_output 

class MLP(torch.nn.Module):
    
    def __init__(
        self, 
        layers_hidden,
        base_activation=torch.nn.SiLU,
    ):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        #self.drop = torch.nn.Dropout(p=0.1) # dropout
        
        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                MLPLayer(
                    input_dim,
                    output_dim,
                    base_activation=base_activation,
                )
            )
    
    def forward(self, x: torch.Tensor):
        #x = self.drop(x)
        for layer in self.layers: 
            x = layer(x)
        return x
        
