"""
GPT-2 Transformer Block with Tensor Parallelism Support

Block Structure:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        GPT-2 TRANSFORMER BLOCK                          │
    │                                                                         │
    │  Input: x [B, T, 768]                                                   │
    │         │                                                               │
    │         ├──────────────────────────────┐                                │
    │         │                              │ (residual)                     │
    │         ▼                              │                                │
    │  ┌─────────────┐                       │                                │
    │  │ LayerNorm 1 │  ln_1.weight/bias     │                                │
    │  └─────────────┘                       │                                │
    │         │                              │                                │
    │         ▼                              │                                │
    │  ┌─────────────┐                       │                                │
    │  │  Attention  │  c_attn + c_proj      │                                │
    │  │    (TP)     │  with AllReduce       │                                │
    │  └─────────────┘                       │                                │
    │         │                              │                                │
    │         ▼                              │                                │
    │       (+)◄─────────────────────────────┘                                │
    │         │                                                               │
    │         ├──────────────────────────────┐                                │
    │         │                              │ (residual)                     │
    │         ▼                              │                                │
    │  ┌─────────────┐                       │                                │
    │  │ LayerNorm 2 │  ln_2.weight/bias     │                                │
    │  └─────────────┘                       │                                │
    │         │                              │                                │
    │         ▼                              │                                │
    │  ┌─────────────┐                       │                                │
    │  │     MLP     │  c_fc + c_proj        │                                │
    │  │    (TP)     │  with AllReduce       │                                │
    │  └─────────────┘                       │                                │
    │         │                              │                                │
    │         ▼                              │                                │
    │       (+)◄─────────────────────────────┘                                │
    │         │                                                               │
    │         ▼                                                               │
    │  Output: x [B, T, 768]                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Note: LayerNorms are REPLICATED across all TP ranks (not sharded).
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from .gpt2_attention import GPT2Attention
from .gpt2_mlp import GPT2MLP


class GPT2Block(nn.Module):
    """
    Single GPT-2 Transformer Block with Tensor Parallelism.
    
    Contains:
    - LayerNorm 1 → Attention → Residual
    - LayerNorm 2 → MLP → Residual
    
    Args:
        config: GPT-2 config with n_embd, n_head, etc.
        tp_rank: This GPU's rank in tensor parallel group
        tp_size: Total number of GPUs in tensor parallel group
        tp_group: Process group for tensor parallel communication
        device: Device for this GPU
        ln_1_weight: LayerNorm 1 weight [embed_dim] (replicated)
        ln_1_bias: LayerNorm 1 bias [embed_dim] (replicated)
        ln_2_weight: LayerNorm 2 weight [embed_dim] (replicated)
        ln_2_bias: LayerNorm 2 bias [embed_dim] (replicated)
        attn_c_attn_weight: Attention c_attn weight (pre-sharded)
        attn_c_attn_bias: Attention c_attn bias (pre-sharded)
        attn_c_proj_weight: Attention c_proj weight (pre-sharded)
        attn_c_proj_bias: Attention c_proj bias (only on tp_rank=0)
        mlp_c_fc_weight: MLP c_fc weight (pre-sharded)
        mlp_c_fc_bias: MLP c_fc bias (pre-sharded)
        mlp_c_proj_weight: MLP c_proj weight (pre-sharded)
        mlp_c_proj_bias: MLP c_proj bias (only on tp_rank=0)
    """
    
    def __init__(
        self,
        config,
        tp_rank: int,
        tp_size: int,
        tp_group: dist.ProcessGroup,
        device: torch.device,
        # LayerNorm weights (replicated, NOT sharded)
        ln_1_weight: torch.Tensor,
        ln_1_bias: torch.Tensor,
        ln_2_weight: torch.Tensor,
        ln_2_bias: torch.Tensor,
        # Attention weights (pre-sharded from distributed_loading)
        attn_c_attn_weight: torch.Tensor,
        attn_c_attn_bias: torch.Tensor,
        attn_c_proj_weight: torch.Tensor,
        attn_c_proj_bias: torch.Tensor,
        # MLP weights (pre-sharded from distributed_loading)
        mlp_c_fc_weight: torch.Tensor,
        mlp_c_fc_bias: torch.Tensor,
        mlp_c_proj_weight: torch.Tensor,
        mlp_c_proj_bias: torch.Tensor,
    ):
        super().__init__()
        
        embed_dim = config.n_embd      # 768
        hidden_dim = config.n_inner or 4 * embed_dim  # 3072
        
        # ═══════════════════════════════════════════════════════════════════
        # LayerNorm 1: Before attention (REPLICATED on all TP ranks)
        # ═══════════════════════════════════════════════════════════════════
        self.ln_1 = nn.LayerNorm(embed_dim, device=device)
        with torch.no_grad():
            self.ln_1.weight.copy_(ln_1_weight)
            self.ln_1.bias.copy_(ln_1_bias)
        
        # ═══════════════════════════════════════════════════════════════════
        # Attention: Multi-head self-attention with TP
        # ═══════════════════════════════════════════════════════════════════
        self.attn = GPT2Attention(
            config=config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            device=device,
            c_attn_weight=attn_c_attn_weight,
            c_attn_bias=attn_c_attn_bias,
            c_proj_weight=attn_c_proj_weight,
            c_proj_bias=attn_c_proj_bias,
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # LayerNorm 2: Before MLP (REPLICATED on all TP ranks)
        # ═══════════════════════════════════════════════════════════════════
        self.ln_2 = nn.LayerNorm(embed_dim, device=device)
        with torch.no_grad():
            self.ln_2.weight.copy_(ln_2_weight)
            self.ln_2.bias.copy_(ln_2_bias)
        
        # ═══════════════════════════════════════════════════════════════════
        # MLP: Feed-forward network with TP
        # ═══════════════════════════════════════════════════════════════════
        self.mlp = GPT2MLP(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            device=device,
            c_fc_weight=mlp_c_fc_weight,
            c_fc_bias=mlp_c_fc_bias,
            c_proj_weight=mlp_c_proj_weight,
            c_proj_bias=mlp_c_proj_bias,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer block.
        
        Args:
            hidden_states: Input [batch, seq_len, embed_dim]
        
        Returns:
            hidden_states: Output [batch, seq_len, embed_dim]
        """
        # ─────────────────────────────────────────────────────────────────
        # Attention sub-block with residual connection
        # x = x + Attention(LayerNorm(x))
        # ─────────────────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)      # [B, T, 768]
        hidden_states = self.attn(hidden_states)      # [B, T, 768] (AllReduced)
        hidden_states = residual + hidden_states      # Residual connection
        
        # ─────────────────────────────────────────────────────────────────
        # MLP sub-block with residual connection
        # x = x + MLP(LayerNorm(x))
        # ─────────────────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)      # [B, T, 768]
        hidden_states = self.mlp(hidden_states)       # [B, T, 768] (AllReduced)
        hidden_states = residual + hidden_states      # Residual connection
        
        return hidden_states