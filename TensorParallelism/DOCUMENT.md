# QuintNet TensorParallelism - Complete Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Environment Setup](#installation--environment-setup)
3. [Core Concepts](#core-concepts)
4. [Architecture Deep Dive](#architecture-deep-dive)
5. [API Reference](#api-reference)
6. [Advanced Usage Patterns](#advanced-usage-patterns)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Examples & Tutorials](#examples--tutorials)
10. [Development Guide](#development-guide)

---

## Introduction

QuintNet TensorParallelism is a production-grade library for implementing tensor parallelism in PyTorch. It provides efficient, scalable solutions for distributing large neural networks across multiple GPUs with minimal code changes.

### What is Tensor Parallelism?

Tensor parallelism splits individual layers across multiple devices, complementing data parallelism by distributing model weights rather than just data batches.

```
Data Parallelism:     Model Parallelism:      Tensor Parallelism:
┌─────┬─────┬─────┐   ┌─────┐ ┌─────┐        ┌───┬───┬───┬───┐
│  M  │  M  │  M  │   │  L1 │ │  L2 │   →    │ L │ L │ L │ L │
│  o  │  o  │  o  │   │  L2 │ │  L3 │        │ 1 │ 1 │ 1 │ 1 │
│  d  │  d  │  d  │   │  L3 │ │  L4 │        └───┴───┴───┴───┘
│  e  │  e  │  e  │   └─────┘ └─────┘        Split within layer
│  l  │  l  │  l  │   Sequential             Parallel execution
└─────┴─────┴─────┘   execution
Different data        Different layers        Same layer split
```

### When to Use Tensor Parallelism

**✅ Good for:**
- Models too large for single GPU memory
- Layers with large weight matrices (transformers, large MLPs)
- High memory requirements with moderate compute
- Research requiring fine-grained parallelism control

**❌ Less suitable for:**
- Small models that fit on single GPU
- Communication-bound scenarios
- Models with many small layers
- Simple data parallel scenarios

---

## Installation & Environment Setup

### Prerequisites

```bash
# Required dependencies
Python >= 3.8
PyTorch >= 2.0.0
CUDA >= 11.0 (for GPU support)
NCCL >= 2.7.0 (for multi-GPU communication)
```

### Environment Setup

```bash
# 1. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Verify CUDA setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# 3. Test distributed setup
python -c "import torch.distributed as dist; print('Distributed available')"
```

### Project Structure Integration

```
YourProject/
├── QuintNet/
│   ├── TensorParallelism/
│   │   ├── __init__.py          # Clean imports
│   │   ├── comm_ops.py          # Communication operations
│   │   ├── layers.py            # Parallel layer implementations
│   │   ├── rewrite.py           # Model conversion utilities
│   │   ├── processgroup.py      # Process group management
│   │   └── utils.py             # Helper utilities (optional)
│   └── __init__.py
├── YourModule/
│   ├── training_script.py
│   └── model.py
└── requirements.txt
```

---

## Core Concepts

### 1. Process Groups and Ranks

```python
# Understanding ranks in tensor parallelism
Total GPUs: 8, TP Size: 4

TP Group 0: [Rank 0, Rank 1, Rank 2, Rank 3]
TP Group 1: [Rank 4, Rank 5, Rank 6, Rank 7]

Each TP group works on the same model split across 4 GPUs
Different TP groups can process different data (like data parallelism)
```

```python
from QuintNet.TensorParallelism import ProcessGroupManager

# Initialize process groups
pgm = ProcessGroupManager(tp_size=4)
tp_group = pgm.get_group()        # This rank's TP group
tp_rank = pgm.get_tp_rank()       # Position within TP group (0-3)
tp_world_size = pgm.get_tp_world_size()  # Size of TP group (4)
```

### 2. Communication Patterns

#### Column Parallelism
```python
# Original layer: Linear(input_dim=1000, output_dim=4000)
# Split across 4 GPUs:

GPU 0: Linear(1000, 1000)  # Outputs columns 0-999
GPU 1: Linear(1000, 1000)  # Outputs columns 1000-1999  
GPU 2: Linear(1000, 1000)  # Outputs columns 2000-2999
GPU 3: Linear(1000, 1000)  # Outputs columns 3000-3999

# Forward: All_Gather to concatenate outputs
# Backward: Slice gradients back to each GPU
```

#### Row Parallelism
```python
# Original layer: Linear(input_dim=4000, output_dim=1000)  
# Split across 4 GPUs:

GPU 0: Linear(1000, 1000)  # Uses input features 0-999
GPU 1: Linear(1000, 1000)  # Uses input features 1000-1999
GPU 2: Linear(1000, 1000)  # Uses input features 2000-2999
GPU 3: Linear(1000, 1000)  # Uses input features 3000-3999

# Forward: All_Reduce to sum partial outputs
# Backward: All_Gather to distribute gradients
```

### 3. Memory and Computation Trade-offs

```python
# Memory usage comparison (approximate)
# Original model: 1 GPU with 16GB model
# With 4-way TP: Each GPU holds ~4GB model weights + activations

Single GPU:    [16GB Model] [8GB Activations] = 24GB total
4-way TP:      [4GB Model]  [8GB Activations] = 12GB per GPU
               ↳ Communication overhead: ~10-20% depending on model
```

---

## Architecture Deep Dive

### Module 1: Communication Operations (`comm_ops.py`)

The foundation of tensor parallelism - efficient distributed communication with autograd support.

#### `All_Gather` Class

**Purpose**: Concatenate tensors from all ranks in a process group.

```python
class All_Gather(torch.autograd.Function):
    """
    Forward: [..., local_dim] → [..., local_dim * world_size]
    Backward: Slice or reduce_scatter based on mode
    """
```

**Detailed Flow:**
```python
# Forward pass example with 4 GPUs
GPU 0: tensor([1, 2])     ┐
GPU 1: tensor([3, 4])     ├─ All_Gather ─→ tensor([1, 2, 3, 4, 5, 6, 7, 8])
GPU 2: tensor([5, 6])     │                (same result on all GPUs)
GPU 3: tensor([7, 8])     ┘

# Backward pass modes:
# Mode "slice": Each GPU gets its gradient slice (fast, no communication)
# Mode "reduce_scatter": Use reduce_scatter (for advanced patterns)
```

**Usage Examples:**
```python
from QuintNet.TensorParallelism import All_Gather

# Basic usage with slice mode (recommended)
gathered = All_Gather.apply(local_tensor, process_group, "slice")

# Advanced usage with reduce_scatter mode
gathered = All_Gather.apply(local_tensor, process_group, "reduce_scatter")
```

#### `All_Reduce` Class

**Purpose**: Sum tensors across all ranks (each rank gets the full sum).

```python
# Forward pass example
GPU 0: tensor([1, 2])     ┐
GPU 1: tensor([3, 4])     ├─ All_Reduce ─→ tensor([10, 12])
GPU 2: tensor([5, 6])     │                (same sum on all GPUs)
GPU 3: tensor([1, 0])     ┘

# Backward: No communication needed (gradient is already replicated)
```

#### `ReduceScatter` Class

**Purpose**: Reduce and scatter - each rank gets a different portion of the reduced result.

```python
# More efficient than All_Reduce + scatter for certain patterns
GPU 0: tensor([1, 2, 3, 4])  ┐              GPU 0: tensor([10])  # Sum of 1st elements
GPU 1: tensor([2, 3, 4, 5])  ├─ReduceScatter GPU 1: tensor([12])  # Sum of 2nd elements  
GPU 2: tensor([3, 4, 5, 6])  │           →   GPU 2: tensor([14])  # Sum of 3rd elements
GPU 3: tensor([4, 5, 6, 7])  ┘              GPU 3: tensor([16])  # Sum of 4th elements
```

### Module 2: Parallel Layers (`layers.py`)

Drop-in replacements for standard PyTorch layers with built-in parallelism.

#### `ColumnParallelLinear` Class

**Architecture:**
```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, 
                 local_device,           # GPU device for this rank
                 tp_group,               # Tensor parallel process group
                 in_features,            # Input dimension (same across ranks)
                 out_features_per_rank,  # Output dim per rank (total_out / tp_size)
                 weight_slice,           # This rank's weight portion
                 bias_slice=None,        # This rank's bias portion (optional)
                 gather_output=True,     # Whether to concatenate outputs
                 sync_gradients=True,    # Whether to sync gradients
                 gather_mode="slice"):   # Backward communication mode
```

**Internal Structure:**
```python
def forward(self, x):
    # Step 1: Ensure input is on correct device
    if x.device != self.device:
        x = x.to(self.device, non_blocking=True)
    
    # Step 2: Local computation
    local_out = self.proj(x)  # Shape: [batch, out_features_per_rank]
    
    # Step 3: Communication (if gather_output=True)
    if self.gather_output:
        return All_Gather.apply(local_out, self.tp_group, self.gather_mode)
        # Output shape: [batch, out_features_per_rank * tp_world_size]
    else:
        return local_out  # Keep local for chaining with row-parallel layers
```

**Use Cases:**
```python
# Standard usage - gather outputs for next layer
col_layer = ColumnParallelLinear(..., gather_output=True)

# Chained with row-parallel layer - no gathering needed
col_layer = ColumnParallelLinear(..., gather_output=False)
row_layer = RowParallelLinear(..., input_is_parallel=True)
```

#### `RowParallelLinear` Class

**Architecture:**
```python
def forward(self, x):
    if self.input_is_parallel:
        # Input already sharded, use directly
        local_out = self.proj(x)
    else:
        # Input is replicated, slice it first
        rank = dist.get_rank(self.tp_group)
        if x.dim() == 3:  # [batch, seq, features]
            start_idx = rank * self.in_features_per_rank
            end_idx = (rank + 1) * self.in_features_per_rank
            inp = x[:, :, start_idx:end_idx]
        # ... handle other dimensions
        
        local_out = self.proj(inp)
    
    # Reduce across ranks to get final output
    output = All_Reduce.apply(local_out, self.tp_group)
    
    # Add bias (only on rank 0 to avoid duplication)
    if self.bias is not None:
        output = output + self.bias
    
    return output
```

#### `VocabParallelEmbedding` Class

**Special considerations for embeddings:**
```python
def forward(self, input_ids):
    # Challenge: Each rank only has part of vocabulary
    # Solution: Mask out tokens not in this rank's range
    
    mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)
    masked_input = (input_ids - self.vocab_start_idx) * mask
    
    # Local embedding lookup (zeros for out-of-range tokens)
    embeddings = self.embedding(masked_input) * mask.unsqueeze(-1).float()
    
    # Sum embeddings across ranks (only one rank contributes per token)
    return All_Reduce.apply(embeddings, self.tp_group)
```

### Module 3: Model Rewriting (`rewrite.py`)

Automatic conversion of standard models to tensor parallel versions.

#### `apply_tensor_parallel` Function

**Algorithm Overview:**
```python
def apply_tensor_parallel(model, tp_size, gather_output=True, 
                         sync_gradients=True, method_of_parallelism="column"):
    # Step 1: Initialize process groups
    pgm = ProcessGroupManager(tp_size)
    tp_group = pgm.get_group()
    
    # Step 2: Get rank information  
    tp_rank = pgm.get_tp_rank()
    tp_world_size = pgm.get_tp_world_size()
    
    # Step 3: Recursively find and replace Linear layers
    def replace_linear(module, path=""):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                # Replace with parallel version
                parallel_layer = create_parallel_layer(child, ...)
                setattr(module, name, parallel_layer)
            else:
                replace_linear(child, f"{path}.{name}")
    
    # Step 4: Execute replacement
    replace_linear(model)
    
    # Step 5: Synchronize all ranks
    dist.barrier()
    
    return model
```

**Weight Splitting Logic:**
```python
# Column parallelism example
original_weight = torch.randn(4000, 1000)  # [out_features, in_features]
tp_size = 4
cols_per_rank = 4000 // 4  # 1000

# Split across ranks
rank_0_weight = original_weight[0:1000, :]     # Rows 0-999
rank_1_weight = original_weight[1000:2000, :]  # Rows 1000-1999
rank_2_weight = original_weight[2000:3000, :]  # Rows 2000-2999
rank_3_weight = original_weight[3000:4000, :]  # Rows 3000-3999
```

**Error Handling:**
```python
# Dimension compatibility checks
if out_features % tp_world_size != 0:
    print(f"Warning: {layer_path} out_features {out_features} "
          f"not divisible by tp_size {tp_world_size}")
    continue  # Skip this layer

# Device placement verification
assert weight_slice.device == local_device, "Weight device mismatch"
```

### Module 4: Process Group Management (`processgroup.py`)

Handles the complexity of organizing multiple GPUs into tensor parallel groups.

#### `ProcessGroupManager` Class

**Group Creation Logic:**
```python
def __init__(self, tp_size):
    self.world_size = dist.get_world_size()  # Total number of processes
    self.rank = dist.get_rank()              # This process's global rank
    
    # Validation
    assert self.world_size % tp_size == 0, "world_size must be divisible by tp_size"
    
    # Group assignment
    group_id = self.rank // tp_size
    ranks = list(range(group_id * tp_size, (group_id + 1) * tp_size))
    
    # Create the actual process group
    self.tp_group = dist.new_group(ranks=ranks)
```

**Example with 8 GPUs, tp_size=4:**
```python
# Group 0: Ranks [0, 1, 2, 3]
# Group 1: Ranks [4, 5, 6, 7]

# For rank 2:
group_id = 2 // 4 = 0
ranks = [0, 1, 2, 3]
# This rank belongs to group 0 and has tp_rank=2 within that group
```

**Multi-Dimensional Parallelism Support:**
```python
# Future extension: Support for data + tensor parallelism
# Example: 16 GPUs, tp_size=4, dp_size=4
# TP Group 0: [0,1,2,3],   TP Group 1: [4,5,6,7]
# TP Group 2: [8,9,10,11], TP Group 3: [12,13,14,15]
# 
# DP across groups: [0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]
```

---

## API Reference

### Core Functions

#### `apply_tensor_parallel(model, tp_size, **kwargs)`

**Purpose**: Convert a standard PyTorch model to tensor parallel version.

**Parameters:**
- `model` (nn.Module): Model to convert
- `tp_size` (int): Tensor parallel size (must divide total GPU count)
- `gather_output` (bool, default=True): Whether to gather layer outputs
- `sync_gradients` (bool, default=True): Whether to synchronize gradients
- `method_of_parallelism` (str, default="column"): "column" or "row"

**Returns:** Modified model with parallel layers

**Example:**
```python
from QuintNet.TensorParallelism import apply_tensor_parallel
import torch.nn as nn

# Original model
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.Linear(256, 10)
)

# Convert to tensor parallel
model = apply_tensor_parallel(
    model=model,
    tp_size=4,
    gather_output=True,
    sync_gradients=True,
    method_of_parallelism="column"
)

# Model interface unchanged - use normally
output = model(input_tensor)
```

### Layer Classes

#### `ColumnParallelLinear`

**Constructor Parameters:**
```python
ColumnParallelLinear(
    local_device: torch.device,          # GPU device
    tp_group: dist.ProcessGroup,         # Process group
    in_features: int,                    # Input feature dimension
    out_features_per_rank: int,          # Output features per rank
    weight_slice: torch.Tensor,          # Weight partition for this rank
    bias_slice: Optional[torch.Tensor],  # Bias partition (optional)
    gather_output: bool = True,          # Whether to gather outputs
    sync_gradients: bool = True,         # Whether to sync gradients
    gather_mode: str = "slice"           # Backward mode ("slice" or "reduce_scatter")
)
```

**Methods:**
- `forward(x)`: Forward pass with automatic communication

#### `RowParallelLinear`

**Constructor Parameters:**
```python
RowParallelLinear(
    local_device: torch.device,
    tp_group: dist.ProcessGroup,
    in_features_per_rank: int,           # Input features per rank
    out_features: int,                   # Output feature dimension
    weight_slice: torch.Tensor,
    bias_slice: Optional[torch.Tensor] = None,
    input_is_parallel: bool = True       # Whether input is pre-sharded
)
```

#### `VocabParallelEmbedding`

**Constructor Parameters:**
```python
VocabParallelEmbedding(
    local_device: torch.device,
    tp_group: dist.ProcessGroup,
    num_embeddings: int,                 # Total vocabulary size
    embedding_dim: int,                  # Embedding dimension
    vocab_start_idx: int,                # Start of this rank's vocab range
    vocab_end_idx: int,                  # End of this rank's vocab range
    weight_slice: torch.Tensor           # Embedding weights for this rank
)
```

### Communication Operations

#### `All_Gather.apply(tensor, group, mode="slice")`

**Parameters:**
- `tensor`: Input tensor to gather
- `group`: Process group for communication
- `mode`: Backward mode ("slice" or "reduce_scatter")

**Returns:** Concatenated tensor from all ranks

#### `All_Reduce.apply(tensor, group)`

**Parameters:**
- `tensor`: Input tensor to reduce
- `group`: Process group for communication

**Returns:** Sum-reduced tensor (same on all ranks)

#### `ReduceScatter.apply(tensor, group)`

**Parameters:**
- `tensor`: Input tensor to reduce and scatter
- `group`: Process group for communication

**Returns:** Reduced tensor chunk for this rank

### Utility Classes

#### `ProcessGroupManager(tp_size)`

**Methods:**
- `get_group()`: Returns the tensor parallel process group
- `get_tp_rank()`: Returns rank within TP group
- `get_tp_world_size()`: Returns TP group size

---

## Examples & Tutorials

### Tutorial 1: Basic MNIST with Tensor Parallelism

