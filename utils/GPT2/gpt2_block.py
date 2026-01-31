
"""
This Contains the implementation of GPT2 Block with Tensor Parallelism Support
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from .gpt2_attention import GPT2Attention
from .gpt2_mlp import GPT2MLP

class GPT2Block(nn.Module):
    def __init__(self, 
                embed_dim, 
                hidden_dim, 
                tp_rank, 
                tp_size, 
                tp_group, 
                device, 
                c_attn_weight, 
                c_attn_bias, 
                c_proj_weight, 
                c_proj_bias, 
                c_fc_weight, 
                c_fc_bias, 
                c_proj_weight, 
                c_proj_bias, 
                dropout_prob
            ):
        super().__init__()
        self.attn = GPT2Attention(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            device=device,
            c_attn_weight=c_attn_weight,
            c_attn_bias=c_attn_bias,
            c_proj_weight=c_proj_weight,
            c_proj_bias=c_proj_bias,
            dropout_prob=dropout_prob
        )
        self.mlp = GPT2MLP(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            device=device,
            c_fc_weight=c_fc_weight,
            c_fc_bias=c_fc_bias,
            c_proj_weight=c_proj_weight,
            c_proj_bias=c_proj_bias,
            dropout_prob=dropout_prob
        )
    
    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x