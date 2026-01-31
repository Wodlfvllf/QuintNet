"""
GPT-2 Pipeline Stage with Tensor Parallelism Support

A GPT2Stage represents one segment of the GPT-2 model for Pipeline Parallelism.
Each PP rank owns exactly one stage containing a subset of transformer blocks.

Stage Structure by PP rank:
┌─────────────────────────────────────────────────────────────────────────────┐
│  PP=2 Example (12 blocks total)                                             │
│                                                                             │
│  Stage 0 (pp_rank=0):                  Stage 1 (pp_rank=1):                 │
│  ├── embedding (wte + wpe)             ├── blocks[6..11]                    │
│  ├── blocks[0..5]                      └── ln_f (final layer norm)          │
│  └── (no ln_f)                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  PP=4 Example (12 blocks total)                                             │
│                                                                             │
│  Stage 0:        Stage 1:        Stage 2:        Stage 3:                   │
│  ├── embedding   ├── blocks[3..5]├── blocks[6..8]├── blocks[9..11]          │
│  ├── blocks[0..2]                                └── ln_f                   │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from .gpt2_embeddings import GPT2Embedding
from .gpt2_block import GPT2Block


class GPT2Stage(nn.Module):
    """
    One pipeline stage of GPT-2.
    
    Contains a subset of transformer blocks, plus optionally:
    - Embeddings (only first stage, pp_rank=0)
    - Final LayerNorm (only last stage, pp_rank=pp_size-1)
    """
    
    def __init__(
        self, 
        embedding: GPT2Embedding,
        blocks: nn.ModuleList,
        ln_f: nn.LayerNorm,
        is_first_stage: bool,
        is_last_stage: bool,
    ):
        super().__init__()
        
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage
        
        # Conditionally set components
        self.embedding = embedding if is_first_stage else None
        self.blocks = blocks
        self.ln_f = ln_f if is_last_stage else None
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for this pipeline stage.
        
        Args:
            x: For first stage: input_ids [B, T]
               For other stages: hidden_states [B, T, embed_dim]
            position_ids: Position IDs [B, T] (only used by first stage)
        
        Returns:
            hidden_states: [B, T, embed_dim]
        """
        # ─────────────────────────────────────────────────────────────────
        # First stage: Apply embeddings to convert token IDs to vectors
        # ─────────────────────────────────────────────────────────────────
        if self.is_first_stage:
            x = self.embedding(x, position_ids)  # [B, T] → [B, T, 768]
        
        # ─────────────────────────────────────────────────────────────────
        # All stages: Process through this stage's transformer blocks
        # ─────────────────────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)  # [B, T, 768] → [B, T, 768]
        
        # ─────────────────────────────────────────────────────────────────
        # Last stage: Apply final layer norm
        # ─────────────────────────────────────────────────────────────────
        if self.is_last_stage:
            x = self.ln_f(x)  # [B, T, 768] → [B, T, 768]
        
        return x
    
    @classmethod
    def from_sharded_state_dict(
        cls,
        state_dict: dict,
        config,
        pp_rank: int,
        pp_size: int,
        tp_rank: int,
        tp_size: int,
        tp_group: dist.ProcessGroup,
        device: torch.device,
    ) -> "GPT2Stage":
        """
        Factory method to build a GPT2Stage from pre-sharded weights.
        
        The state_dict should come from load_gpt2_distributed(), which already
        contains only the weights needed for this GPU.
        
        Args:
            state_dict: Pre-sharded weights from load_gpt2_distributed()
            config: GPT-2 config (n_embd, n_head, n_layer, etc.)
            pp_rank: This GPU's pipeline parallel rank
            pp_size: Total pipeline parallel size
            tp_rank: This GPU's tensor parallel rank
            tp_size: Total tensor parallel size
            tp_group: Tensor parallel process group
            device: Device for this GPU
        
        Returns:
            GPT2Stage instance with loaded weights
        """
        is_first_stage = (pp_rank == 0)
        is_last_stage = (pp_rank == pp_size - 1)
        
        embed_dim = config.n_embd
        num_blocks = config.n_layer
        blocks_per_stage = num_blocks // pp_size
        start_block = pp_rank * blocks_per_stage
        end_block = start_block + blocks_per_stage
        
        # ═══════════════════════════════════════════════════════════════════
        # Build Embedding (only first stage)
        # ═══════════════════════════════════════════════════════════════════
        embedding = None
        if is_first_stage:
            embedding = GPT2Embedding(
                vocab_size=config.vocab_size,
                max_position_embeddings=config.n_positions,
                embed_dim=embed_dim,
                wte_weights=state_dict['wte.weight'],
                wpe_weights=state_dict['wpe.weight'],
                dropout_prob=config.embd_pdrop,
                device=device,
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # Build Transformer Blocks
        # ═══════════════════════════════════════════════════════════════════
        blocks = []
        for block_idx in range(start_block, end_block):
            prefix = f"h.{block_idx}"
            
            block = GPT2Block(
                config=config,
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_group=tp_group,
                device=device,
                # LayerNorm weights (replicated)
                ln_1_weight=state_dict[f'{prefix}.ln_1.weight'],
                ln_1_bias=state_dict[f'{prefix}.ln_1.bias'],
                ln_2_weight=state_dict[f'{prefix}.ln_2.weight'],
                ln_2_bias=state_dict[f'{prefix}.ln_2.bias'],
                # Attention weights (pre-sharded)
                attn_c_attn_weight=state_dict[f'{prefix}.attn.c_attn.weight'],
                attn_c_attn_bias=state_dict[f'{prefix}.attn.c_attn.bias'],
                attn_c_proj_weight=state_dict[f'{prefix}.attn.c_proj.weight'],
                attn_c_proj_bias=state_dict.get(f'{prefix}.attn.c_proj.bias'),
                # MLP weights (pre-sharded)
                mlp_c_fc_weight=state_dict[f'{prefix}.mlp.c_fc.weight'],
                mlp_c_fc_bias=state_dict[f'{prefix}.mlp.c_fc.bias'],
                mlp_c_proj_weight=state_dict[f'{prefix}.mlp.c_proj.weight'],
                mlp_c_proj_bias=state_dict.get(f'{prefix}.mlp.c_proj.bias'),
            )
            blocks.append(block)
        
        blocks = nn.ModuleList(blocks)
        
        # ═══════════════════════════════════════════════════════════════════
        # Build Final LayerNorm (only last stage)
        # ═══════════════════════════════════════════════════════════════════
        ln_f = None
        if is_last_stage:
            ln_f = nn.LayerNorm(embed_dim, device=device)
            with torch.no_grad():
                ln_f.weight.copy_(state_dict['ln_f.weight'])
                ln_f.bias.copy_(state_dict['ln_f.bias'])
        
        # ═══════════════════════════════════════════════════════════════════
        # Create and return the stage instance
        # This calls __init__(embedding, blocks, ln_f, is_first, is_last)
        # ═══════════════════════════════════════════════════════════════════
        return cls(
            embedding=embedding,
            blocks=blocks,
            ln_f=ln_f,
            is_first_stage=is_first_stage,
            is_last_stage=is_last_stage,
        )