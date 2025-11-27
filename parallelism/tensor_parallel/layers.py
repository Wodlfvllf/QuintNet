"""
Tensor Parallel Layer Implementations

This module provides specialized `torch.nn.Module` implementations for various
tensor parallelism strategies. These layers are designed to be used in conjunction
with a model rewriting mechanism that replaces standard `nn.Linear` or `nn.Embedding`
layers with their tensor-parallel counterparts.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

Tensor parallelism (TP) involves sharding the weights of individual layers
across multiple devices. This allows very large layers, which might not fit
on a single GPU, to be distributed. The key challenge lies in orchestrating
the communication (e.g., all-gather, all-reduce) required to ensure that
forward and backward passes produce correct results despite the sharded weights.

-   **`ColumnParallelLinear`**: Splits the `out_features` dimension of a
    linear layer. Each device computes a partial output, which may then be
    all-gathered to form the complete output.
-   **`RowParallelLinear`**: Splits the `in_features` dimension of a linear
    layer. Each device contributes a partial sum, which is then all-reduced
    to form the complete output.
-   **`VocabParallelEmbedding`**: Splits the vocabulary dimension of an
    embedding layer.

These layers rely on custom autograd functions in `QuintNet.core.communication`
to handle the collective communication operations during both forward and
backward passes.

===============================================================================
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from ...core.communication import All_Gather, All_Reduce
from typing import Optional, List, Tuple


class ColumnParallelLinear(nn.Module):
    """
    Implements a column-wise sharded linear layer for Tensor Parallelism.

    This layer shards the output features (`out_features`) across the tensor
    parallel group. Each device computes a portion of the output. Depending
    on `gather_output`, the outputs from all devices can be concatenated
    into a full output, or only the local shard is returned.
    """

    def __init__(self, local_device: torch.device, tp_group: dist.ProcessGroup,
                 in_features: int, out_features_per_rank: int,
                 weight_slice: torch.Tensor, bias_slice: Optional[torch.Tensor] = None,
                 gather_output: bool = True, sync_gradients: bool = True,
                 gather_mode: str = "slice"):
        """
        Initializes the ColumnParallelLinear layer.

        Args:
            local_device (torch.device): The device the layer should reside on.
            tp_group (dist.ProcessGroup): The tensor parallelism communication group.
            in_features (int): The number of input features to the linear layer.
            out_features_per_rank (int): The number of output features for
                the local shard on this rank.
            weight_slice (torch.Tensor): The sharded weight tensor for this rank.
            bias_slice (Optional[torch.Tensor]): The sharded bias tensor for this rank.
            gather_output (bool): If True, all-gathers the output from all ranks
                to form the complete output. If False, returns only the local shard.
            sync_gradients (bool): If True, gradients are synchronized during backward pass.
                (Note: this is handled by custom autograd functions).
            gather_mode (str): The method to use for gathering ('slice' or 'reduce_scatter').
        """
        super().__init__()
        self.device = local_device
        self.tp_group = tp_group
        self.gather_output = gather_output
        self.sync_gradients = sync_gradients # Currently handled by All_Gather's autograd
        self.gather_mode = gather_mode  # "slice" or "reduce_scatter"

        # Create a local nn.Linear module with the sharded output features
        self.proj = nn.Linear(in_features, out_features_per_rank, bias=(bias_slice is not None),
                              device=self.device)

        # Copy the pre-sharded weights and biases into the layer, disabling grad for copy
        with torch.no_grad():
            self.proj.weight.copy_(weight_slice.to(self.device))
            if bias_slice is not None:
                self.proj.bias.copy_(bias_slice.to(self.device))

        # Ensure gradients are tracked for optimization
        self.proj.weight.requires_grad_(True)
        if self.proj.bias is not None:
            self.proj.bias.requires_grad_(True)

        # Gradients for parallel layers are managed by custom autograd functions
        # for communication primitives (e.g., All_Gather, All_Reduce) to ensure
        # efficient collective operations during the backward pass.
        # Thus, we do not register explicit backward hooks here.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the ColumnParallelLinear layer.

        Args:
            x (torch.Tensor): The input tensor (could be replicated or gathered
                from previous layers).

        Returns:
            torch.Tensor: The output tensor, either local or all-gathered,
                depending on `gather_output`.
        """
        rank = dist.get_rank()
        print(f"[Rank {rank}] ColumnParallelLinear: START forward", flush=True)

        # Move input to the local device if needed
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        # Local computation of the partial output
        local_out = self.proj(x)  # shape (B, out_features_per_rank)

        if self.gather_output:
            # If `gather_output` is True, combine outputs from all ranks.
            # The custom `All_Gather.apply` function handles both forward
            # communication and backward gradient synchronization.
            print(f"[Rank {rank}] ColumnParallelLinear: Calling All_Gather", flush=True)
            res = All_Gather.apply(local_out, self.tp_group, self.gather_mode)
            print(f"[Rank {rank}] ColumnParallelLinear: Finished All_Gather", flush=True)
            return res
        else:
            # If `gather_output` is False, return only the local shard.
            # This is typically used when the next layer is `RowParallelLinear`
            # and expects sharded input.
            print(f"[Rank {rank}] ColumnParallelLinear: END forward (no gather)", flush=True)
            return local_out


class RowParallelLinear(nn.Module):
    """
    Implements a row-wise sharded linear layer for Tensor Parallelism.

    This layer shards the input features (`in_features`) across the tensor
    parallel group. It expects either sharded input from a prior
    `ColumnParallelLinear` layer (with `gather_output=False`) or a replicated
    input that needs to be locally sliced. The outputs from all ranks are
    all-reduced to produce the final, complete output.
    """
    def __init__(self, local_device: torch.device, tp_group: dist.ProcessGroup,
                 in_features_per_rank: int, out_features: int,
                 weight_slice: torch.Tensor, bias_slice: Optional[torch.Tensor] = None,
                 input_is_parallel: bool = True):
        """
        Initializes the RowParallelLinear layer.

        Args:
            local_device (torch.device): The device the layer should reside on.
            tp_group (dist.ProcessGroup): The tensor parallelism communication group.
            in_features_per_rank (int): The number of input features for the
                local shard on this rank.
            out_features (int): The total number of output features for the layer.
            weight_slice (torch.Tensor): The sharded weight tensor for this rank.
            bias_slice (Optional[torch.Tensor]): The bias tensor for the layer.
                (Note: only applied on rank 0 of the TP group to avoid duplication).
            input_is_parallel (bool): If True, assume input `x` is already
                sharded across ranks. If False, the input `x` is assumed to be
                replicated, and this layer will slice it internally.
        """
        super().__init__()
        self.device = local_device
        self.tp_group = tp_group
        self.input_is_parallel = input_is_parallel
        self.in_features_per_rank = in_features_per_rank

        # Local linear layer without bias (bias is handled separately via all-reduce)
        self.proj = nn.Linear(in_features_per_rank, out_features, bias=False, device=self.device)
        
        with torch.no_grad():
            self.proj.weight.copy_(weight_slice.to(self.device))
        
        # The bias is only added on rank 0 of the TP group to avoid summing it N times.
        # `dist.get_rank(self.tp_group)` gets the rank *within* the TP group.
        if bias_slice is not None and dist.get_rank(self.tp_group) == 0:
            self.bias = nn.Parameter(bias_slice.to(self.device))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the RowParallelLinear layer.

        Args:
            x (torch.Tensor): The input tensor, either already sharded or replicated.

        Returns:
            torch.Tensor: The all-reduced output tensor.
        """
        rank = dist.get_rank()
        print(f"[Rank {rank}] RowParallelLinear: START forward", flush=True)
        
        # Move input to the local device if needed
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        
        if self.input_is_parallel:
            # If input is already parallel (e.g., from ColumnParallelLinear), use it directly
            inp_for_proj = x
        else:
            # If input is replicated (e.g., from an embedding layer), slice it for this rank
            rank_in_tp_group = dist.get_rank(self.tp_group)
            if x.dim() == 3: # (Batch, Sequence Length, Features)
                inp_for_proj = x[:, :, rank_in_tp_group * self.in_features_per_rank : (rank_in_tp_group + 1) * self.in_features_per_rank]
            elif x.dim() == 2: # (Batch, Features)
                inp_for_proj = x[:, rank_in_tp_group * self.in_features_per_rank : (rank_in_tp_group + 1) * self.in_features_per_rank]
            else:
                raise ValueError(f"Unexpected input shape for RowParallelLinear (input_is_parallel=False): {x.shape}")
            
        local_out = self.proj(inp_for_proj)
        
        # All-reduce partial outputs from all ranks to get the complete sum
        # The custom `All_Reduce.apply` function handles both forward
        # communication and backward gradient synchronization.
        print(f"[Rank {rank}] RowParallelLinear: Calling All_Reduce", flush=True)
        local_out = All_Reduce.apply(local_out, self.tp_group)
        print(f"[Rank {rank}] RowParallelLinear: Finished All_Reduce", flush=True)
        
        if self.bias is not None:
            local_out = local_out + self.bias
            
        print(f"[Rank {rank}] RowParallelLinear: END forward", flush=True)
        return local_out


class VocabParallelEmbedding(nn.Module):
    """
    Implements a vocabulary-wise sharded embedding layer for Tensor Parallelism.

    This layer shards the vocabulary dimension of the embedding table across
    the tensor parallel group. Each rank is responsible for a subset of the
    vocabulary. During the forward pass, if an `input_id` falls within a
    rank's vocabulary range, that rank computes its embedding; otherwise,
    it computes zeros. The results are then all-reduced to get the final,
    correct embedding.
    """
    def __init__(self, local_device: torch.device, tp_group: dist.ProcessGroup,
                 num_embeddings: int, embedding_dim: int, 
                 vocab_start_idx: int, vocab_end_idx: int, weight_slice: torch.Tensor):
        """
        Initializes the VocabParallelEmbedding layer.

        Args:
            local_device (torch.device): The device the layer should reside on.
            tp_group (dist.ProcessGroup): The tensor parallelism communication group.
            num_embeddings (int): The total size of the vocabulary.
            embedding_dim (int): The dimension of the embedding vectors.
            vocab_start_idx (int): The starting index of the vocabulary shard
                assigned to this rank.
            vocab_end_idx (int): The exclusive ending index of the vocabulary shard
                assigned to this rank.
            weight_slice (torch.Tensor): The sharded weight tensor for this rank.
        """
        super().__init__()
        self.device = local_device
        self.tp_group = tp_group
        self.vocab_start_idx = vocab_start_idx
        self.vocab_end_idx = vocab_end_idx
        
        # Calculate the local vocabulary size for this rank
        local_vocab_size = vocab_end_idx - vocab_start_idx
        # Create a standard Embedding layer for the local vocabulary shard
        self.embedding = nn.Embedding(local_vocab_size, embedding_dim, device=self.device)
        
        with torch.no_grad():
            self.embedding.weight.copy_(weight_slice.to(self.device))
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the VocabParallelEmbedding layer.

        Args:
            input_ids (torch.Tensor): A tensor of token IDs to embed.

        Returns:
            torch.Tensor: The all-reduced embedding vectors.
        """
        rank = dist.get_rank()
        print(f"[Rank {rank}] VocabParallelEmbedding: START forward", flush=True)
        
        # Ensure input on proper device
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device, non_blocking=True)

        # Create a mask to identify tokens that belong to this rank's vocabulary shard
        mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)
        
        # Shift input_ids to be relative to this rank's local vocabulary start
        masked_input = input_ids - self.vocab_start_idx
        # Zero out input_ids that are not in this rank's vocabulary
        masked_input = masked_input * mask
        
        embeddings = self.embedding(masked_input)
        # Zero out embeddings for tokens not in this rank's vocabulary shard
        embeddings = embeddings * mask.unsqueeze(-1).float()
        
        # All-reduce to combine embeddings from all ranks. Tokens not in a rank's
        # vocabulary will have zero embeddings from that rank, so the all-reduce
        # effectively gathers the correct embedding from the responsible rank.
        print(f"[Rank {rank}] VocabParallelEmbedding: Calling All_Reduce", flush=True)
        res = All_Reduce.apply(embeddings, self.tp_group)
        print(f"[Rank {rank}] VocabParallelEmbedding: Finished All_Reduce", flush=True)
        return res
