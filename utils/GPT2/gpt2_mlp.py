

"""
GPT-2 MLP Implementation
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from ...parallelism import RowParallelLinear, ColumnParallelLinear


class GPT2MLP(nn.Module):
    def __init__(
        self,
        input_dim : int,
        hidden_dim : int,
        output_dim : int,
        activation : nn.Module = nn.GELU(),
        dropout_prob : float = 0.1,
        device : torch.device = None,
        c_fc_weights : torch.Tensor = None,
        c_fc_bias : torch.Tensor = None,
        c_proj_weights : torch.Tensor = None,
        c_proj_bias : torch.Tensor = None,
        tp_rank : int = 0,
        tp_size : int = 1,
        tp_group : dist.ProcessGroup = None
        ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_prob = dropout_prob
        self.device = device
        self.c_fc_weights = c_fc_weights
        self.c_fc_bias = c_fc_bias
        self.c_proj_weights = c_proj_weights
        self.c_proj_bias = c_proj_bias
        
        self.local_input_dim = input_dim // tp_size
        self.local_output_dim = hidden_dim // tp_size
        
        self.c_fc = ColumnParallelLinear(
                                    in_features=self.input_dim, 
                                    out_features_per_rank=self.local_output_dim, 
                                    device=device,
                                    tp_group=tp_group,
                                    weight_slice=c_fc_weights,
                                    bias_slice=c_fc_bias,
                                    gather_output=False
                                    )
        self.c_proj = RowParallelLinear(
                                    in_features=self.local_output_dim, 
                                    out_features_per_rank=self.output_dim, 
                                    device=device,
                                    tp_group=tp_group,
                                    weight_slice=c_proj_weights,
                                    bias_slice=c_proj_bias,
                                    input_is_parallel=True
                                    )
    
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x