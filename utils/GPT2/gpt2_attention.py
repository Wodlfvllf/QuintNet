"""
GPT-2 Attention with Tensor Parallelism Support

This module implements the multi-head self-attention mechanism for GPT-2,
with built-in support for Tensor Parallelism (TP).

Key Design:
- c_attn (Q, K, V projection): Column Parallel - shards output dimension
- c_proj (output projection): Row Parallel - shards input dimension, AllReduce after

With TP=2 and 12 heads:
- GPU 0: heads 0-5 (c_attn: [768, 1152], c_proj: [384, 768])
- GPU 1: heads 6-11 (c_attn: [768, 1152], c_proj: [384, 768])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ...parallelism import ColumnParallelLinear, RowParallelLinear


class GPT2Attention(nn.Module):
    """
    GPT-2 Multi-Head Attention with Tensor Parallelism.
    
    This implementation uses:
    - Fused QKV projection (c_attn) for efficiency
    - Column parallel for c_attn (shards heads across GPUs)
    - Row parallel for c_proj (each GPU has partial input, AllReduce combines)
    
    Args:
        config: GPT-2 config with n_embd, n_head, attn_pdrop, resid_pdrop
        tp_rank: This GPU's rank in the tensor parallel group
        tp_size: Total number of GPUs in tensor parallel group
        tp_group: The process group for tensor parallel communication
        device: The device (cuda:X) for this GPU
        c_attn_weight: Pre-sharded c_attn weight [n_embd, 3*n_embd/tp_size]
        c_attn_bias: Pre-sharded c_attn bias [3*n_embd/tp_size]
        c_proj_weight: Pre-sharded c_proj weight [n_embd/tp_size, n_embd]
        c_proj_bias: Pre-sharded c_proj bias [n_embd] (only on tp_rank=0)
    """
    
    def __init__(
        self, 
        config, 
        tp_rank: int, 
        tp_size: int, 
        tp_group: dist.ProcessGroup,
        device: torch.device,
        c_attn_weight: torch.Tensor,
        c_attn_bias: torch.Tensor,
        c_proj_weight: torch.Tensor,
        c_proj_bias: torch.Tensor = None,
    ):
        super().__init__()
        
        # Config
        self.embed_dim = config.n_embd           # 768
        self.num_heads = config.n_head           # 12
        self.head_dim = self.embed_dim // self.num_heads  # 64
        
        # Tensor Parallel info
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.device = device
        
        # Local dimensions (per GPU)
        self.num_heads_local = self.num_heads // tp_size    # 12/2 = 6 heads per GPU
        self.embed_dim_local = self.embed_dim // tp_size    # 768/2 = 384
        
        # ═══════════════════════════════════════════════════════════════════
        # c_attn: Column Parallel Linear
        # Projects input to Q, K, V (fused) with output sharded across GPUs
        # Full: [768, 2304] → Sharded: [768, 1152] per GPU
        # Output: [B, T, 1152] containing this GPU's portion of Q, K, V
        # ═══════════════════════════════════════════════════════════════════
        self.c_attn = ColumnParallelLinear(
            local_device=device,
            tp_group=tp_group,
            in_features=self.embed_dim,
            out_features_per_rank=3 * self.embed_dim_local,  # 3 * 384 = 1152
            weight_slice=c_attn_weight,
            bias_slice=c_attn_bias,
            gather_output=False,  # Keep sharded - we'll do local attention
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # c_proj: Row Parallel Linear
        # Projects attention output back to hidden dim with AllReduce
        # Full: [768, 768] → Sharded: [384, 768] per GPU
        # Input: [B, T, 384] (partial from this GPU's heads)
        # Output: [B, T, 768] after AllReduce combines all GPUs
        # ═══════════════════════════════════════════════════════════════════
        self.c_proj = RowParallelLinear(
            local_device=device,
            tp_group=tp_group,
            in_features_per_rank=self.embed_dim_local,  # 384
            out_features=self.embed_dim,                # 768
            weight_slice=c_proj_weight,
            bias_slice=c_proj_bias,  # Only tp_rank=0 has bias
            input_is_parallel=True,  # Input is sharded across GPUs
        )
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Scale factor for attention
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for tensor-parallel attention.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, embed_dim]
                          Same tensor on all TP ranks
        
        Returns:
            output: Attention output [batch, seq_len, embed_dim]
                   Same tensor on all TP ranks (after AllReduce)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # ─────────────────────────────────────────────────────────────────
        # Step 1: Fused QKV projection (Column Parallel)
        # Each GPU computes its portion of Q, K, V
        # ─────────────────────────────────────────────────────────────────
        qkv = self.c_attn(hidden_states)  # [B, T, 1152] = [B, T, 3 * 384]
        
        # ─────────────────────────────────────────────────────────────────
        # Step 2: Split into Q, K, V for THIS GPU's heads
        # ─────────────────────────────────────────────────────────────────
        # Each has shape: [B, T, 384] (= 6 heads × 64 head_dim)
        q, k, v = qkv.split(self.embed_dim_local, dim=-1)
        
        # ─────────────────────────────────────────────────────────────────
        # Step 3: Reshape for multi-head attention
        # [B, T, 384] → [B, T, 6, 64] → [B, 6, T, 64]
        # ─────────────────────────────────────────────────────────────────
        q = q.view(batch_size, seq_len, self.num_heads_local, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads_local, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads_local, self.head_dim)
        
        q = q.transpose(1, 2)  # [B, 6, T, 64]
        k = k.transpose(1, 2)  # [B, 6, T, 64]
        v = v.transpose(1, 2)  # [B, 6, T, 64]
        
        # ─────────────────────────────────────────────────────────────────
        # Step 4: Compute attention (local, no communication)
        # Using PyTorch 2.0+ scaled_dot_product_attention for efficiency
        # ─────────────────────────────────────────────────────────────────
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,  # Causal mask for autoregressive generation
        )
        # [B, 6, T, 64]
        
        # ─────────────────────────────────────────────────────────────────
        # Step 5: Reshape back to [B, T, 384]
        # ─────────────────────────────────────────────────────────────────
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, T, 6, 64]
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim_local)
        # [B, T, 384]
        
        # ─────────────────────────────────────────────────────────────────
        # Step 6: Output projection (Row Parallel with AllReduce)
        # Each GPU has [B, T, 384], after c_proj + AllReduce → [B, T, 768]
        # ─────────────────────────────────────────────────────────────────
        output = self.c_proj(attn_output)  # [B, T, 768] - AllReduced internally
        
        # ─────────────────────────────────────────────────────────────────
        # Step 7: Residual dropout
        # ─────────────────────────────────────────────────────────────────
        output = self.resid_dropout(output)
        
        return output