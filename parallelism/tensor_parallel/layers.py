"""
Tensor parallel layer implementations.

This module provides parallel versions of common neural network layers
including ColumnParallelLinear, RowParallelLinear, and VocabParallelEmbedding.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from QuintNet.core.communication import All_Gather, All_Reduce


class ColumnParallelLinear(nn.Module):
    """
    Holds a column shard of nn.Linear (split across out_features).
    If gather_output=True, returns the concatenated full output (each rank holds full).
    If gather_output=False, returns local output (for following row-parallel layers).
    """

    def __init__(self, local_device, tp_group, in_features, out_features_per_rank,
                 weight_slice, bias_slice=None, gather_output=True, sync_gradients=True,
                 gather_mode="slice"):
        super().__init__()
        self.device = local_device
        self.tp_group = tp_group
        self.gather_output = gather_output
        self.sync_gradients = sync_gradients
        self.gather_mode = gather_mode  # "slice" or "reduce_scatter"

        # Create a local linear with out_features_per_rank
        self.proj = nn.Linear(in_features, out_features_per_rank, bias=(bias_slice is not None),
                              device=self.device)

        # Copy weights and bias into the parameter (no-grad)
        with torch.no_grad():
            self.proj.weight.copy_(weight_slice.to(self.device))
            if bias_slice is not None:
                self.proj.bias.copy_(bias_slice.to(self.device))

        # Ensure gradients are tracked
        self.proj.weight.requires_grad_(True)
        if self.proj.bias is not None:
            self.proj.bias.requires_grad_(True)

        # Prefer autograd function for comms (no hook), so we do not register backward hooks here.

    def forward(self, x):
        # Ensure input on proper device
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        local_out = self.proj(x)  # shape (B, out_features_per_rank)

        if self.gather_output:
            # IMPORTANT: use .apply to create autograd node
            return All_Gather.apply(local_out, self.tp_group, self.gather_mode)
        else:
            # Useful if next layer expects local shard only
            return local_out


class RowParallelLinear(nn.Module):
    """
    Holds a row shard of an nn.Linear (split along in_features).
    Expects input to be already sharded and reduces output across TP group.
    """
    def __init__(self, local_device, tp_group, in_features_per_rank, out_features,
                 weight_slice, bias_slice=None, input_is_parallel=True):
        super().__init__()
        self.device = local_device
        self.tp_group = tp_group
        self.input_is_parallel = input_is_parallel
        self.in_features_per_rank = in_features_per_rank

        self.proj = nn.Linear(in_features_per_rank, out_features, bias=False, device=self.device)
        
        with torch.no_grad():
            self.proj.weight.copy_(weight_slice.to(self.device))
        
        # Bias only on first rank to avoid duplication
        if bias_slice is not None and dist.get_rank(self.tp_group) == 0:
            self.bias = nn.Parameter(bias_slice.to(self.device))
        else:
            self.bias = None

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        
        if self.input_is_parallel:
            # Input is already sharded, use directly
            local_out = self.proj(x)
        else:
            # Input is replicated, slice it
            rank = dist.get_rank(self.tp_group)
            if x.dim() == 3:
                inp = x[:, :, rank * self.in_features_per_rank : (rank + 1) * self.in_features_per_rank]
            elif x.dim() == 2:
                inp = x[:, rank * self.in_features_per_rank : (rank + 1) * self.in_features_per_rank]
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
            
            local_out = self.proj(inp)
        
        # All-reduce across TP group
        local_out = All_Reduce.apply(local_out, self.tp_group)
        
        if self.bias is not None:
            local_out = local_out + self.bias
            
        return local_out


class VocabParallelEmbedding(nn.Module):
    """Vocabulary-parallel embedding layer"""
    def __init__(self, local_device, tp_group, num_embeddings, embedding_dim, 
                 vocab_start_idx, vocab_end_idx, weight_slice):
        super().__init__()
        self.device = local_device
        self.tp_group = tp_group
        self.vocab_start_idx = vocab_start_idx
        self.vocab_end_idx = vocab_end_idx
        
        local_vocab_size = vocab_end_idx - vocab_start_idx
        self.embedding = nn.Embedding(local_vocab_size, embedding_dim, device=self.device)
        
        with torch.no_grad():
            self.embedding.weight.copy_(weight_slice.to(self.device))
    
    def forward(self, input_ids):
        # Mask out tokens not in this rank's vocabulary range
        mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)
        masked_input = input_ids - self.vocab_start_idx
        masked_input = masked_input * mask
        
        embeddings = self.embedding(masked_input)
        embeddings = embeddings * mask.unsqueeze(-1).float()  # Zero out invalid positions
        
        # All-reduce to combine embeddings from all ranks
        return All_Reduce.apply(embeddings, self.tp_group)