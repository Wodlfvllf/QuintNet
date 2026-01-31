"""
GPT-2 MLP with Tensor Parallelism Support

MLP Architecture in GPT-2:
    Input [B, T, 768] → c_fc [768, 3072] → GELU → c_proj [3072, 768] → Output [B, T, 768]
    
With Tensor Parallelism (TP=2):
    - c_fc: Column Parallel - shards output dimension (3072 → 1536 per GPU)
    - c_proj: Row Parallel - shards input dimension (3072 → 1536 per GPU), AllReduce after

Data Flow with TP=2:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  GPU 0                                    GPU 1                         │
    │                                                                         │
    │  Input: x [B, T, 768]                    Input: x [B, T, 768]           │
    │  (same on both GPUs)                     (same on both GPUs)            │
    │         │                                       │                       │
    │         ▼                                       ▼                       │
    │  c_fc [768, 1536]                        c_fc [768, 1536]               │
    │         │                                       │                       │
    │         ▼                                       ▼                       │
    │  hidden [B, T, 1536]                     hidden [B, T, 1536]            │
    │  (first half)                            (second half)                  │
    │         │                                       │                       │
    │         ▼                                       ▼                       │
    │      GELU                                    GELU                       │
    │         │                                       │                       │
    │         ▼                                       ▼                       │
    │  c_proj [1536, 768]                      c_proj [1536, 768]             │
    │         │                                       │                       │
    │         ▼                                       ▼                       │
    │  partial [B, T, 768]                     partial [B, T, 768]            │
    │         │                                       │                       │
    │         └───────────────┬───────────────────────┘                       │
    │                         │                                               │
    │                    AllReduce                                            │
    │                         │                                               │
    │                         ▼                                               │
    │              Output: [B, T, 768]                                        │
    │              (same on both GPUs)                                        │
    └─────────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from ...parallelism import ColumnParallelLinear, RowParallelLinear


class GPT2MLP(nn.Module):
    """
    GPT-2 MLP (Feed-Forward) with Tensor Parallelism.
    
    Args:
        embed_dim: Input/output embedding dimension (768 for GPT-2)
        hidden_dim: Intermediate hidden dimension (3072 for GPT-2, = 4 * embed_dim)
        tp_rank: This GPU's rank in tensor parallel group
        tp_size: Total number of GPUs in tensor parallel group
        tp_group: Process group for tensor parallel communication
        device: Device for this GPU
        c_fc_weight: Pre-sharded c_fc weight [embed_dim, hidden_dim/tp_size]
        c_fc_bias: Pre-sharded c_fc bias [hidden_dim/tp_size]
        c_proj_weight: Pre-sharded c_proj weight [hidden_dim/tp_size, embed_dim]
        c_proj_bias: c_proj bias [embed_dim] (only on tp_rank=0)
        dropout_prob: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        tp_rank: int,
        tp_size: int,
        tp_group: dist.ProcessGroup,
        device: torch.device,
        c_fc_weight: torch.Tensor,
        c_fc_bias: torch.Tensor,
        c_proj_weight: torch.Tensor,
        c_proj_bias: torch.Tensor = None,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim      # 768
        self.hidden_dim = hidden_dim    # 3072
        self.tp_size = tp_size
        
        # Per-GPU dimensions
        self.hidden_dim_local = hidden_dim // tp_size  # 3072/2 = 1536
        
        # ═══════════════════════════════════════════════════════════════════
        # c_fc: Column Parallel Linear (up projection)
        # Full: [768, 3072] → Sharded: [768, 1536] per GPU
        # Input: [B, T, 768] (full, same on all GPUs)
        # Output: [B, T, 1536] (sharded, different on each GPU)
        # ═══════════════════════════════════════════════════════════════════
        self.c_fc = ColumnParallelLinear(
            local_device=device,
            tp_group=tp_group,
            in_features=embed_dim,                    # 768 (full input)
            out_features_per_rank=self.hidden_dim_local,  # 1536 (sharded output)
            weight_slice=c_fc_weight,                 # [768, 1536]
            bias_slice=c_fc_bias,                     # [1536]
            gather_output=False,                      # Keep sharded for GELU + c_proj
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # c_proj: Row Parallel Linear (down projection)
        # Full: [3072, 768] → Sharded: [1536, 768] per GPU
        # Input: [B, T, 1536] (sharded, from c_fc)
        # Output: [B, T, 768] (full, after AllReduce)
        # ═══════════════════════════════════════════════════════════════════
        self.c_proj = RowParallelLinear(
            local_device=device,
            tp_group=tp_group,
            in_features_per_rank=self.hidden_dim_local,  # 1536 (sharded input)
            out_features=embed_dim,                       # 768 (full output)
            weight_slice=c_proj_weight,                   # [1536, 768]
            bias_slice=c_proj_bias,                       # [768] only on tp_rank=0
            input_is_parallel=True,                       # Input already sharded
        )
        
        # Dropout applied after c_proj
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for tensor-parallel MLP.
        
        Args:
            hidden_states: Input [batch, seq_len, embed_dim]
                          Same on all TP ranks
        
        Returns:
            output: MLP output [batch, seq_len, embed_dim]
                   Same on all TP ranks (after AllReduce)
        """
        # ─────────────────────────────────────────────────────────────────
        # Step 1: Up projection (Column Parallel)
        # [B, T, 768] → [B, T, 1536] (sharded)
        # ─────────────────────────────────────────────────────────────────
        hidden_states = self.c_fc(hidden_states)
        
        # ─────────────────────────────────────────────────────────────────
        # Step 2: Activation (local, no communication)
        # [B, T, 1536] → [B, T, 1536]
        # ─────────────────────────────────────────────────────────────────
        hidden_states = nn.functional.gelu(hidden_states)
        
        # ─────────────────────────────────────────────────────────────────
        # Step 3: Down projection (Row Parallel with AllReduce)
        # [B, T, 1536] → [B, T, 768] (AllReduced)
        # ─────────────────────────────────────────────────────────────────
        hidden_states = self.c_proj(hidden_states)
        
        # ─────────────────────────────────────────────────────────────────
        # Step 4: Dropout
        # ─────────────────────────────────────────────────────────────────
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states