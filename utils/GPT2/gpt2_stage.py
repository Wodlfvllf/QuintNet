"""
GPT-2 Pipeline Stage with Tensor Parallelism Support

A GPT2Stage represents one segment of the GPT-2 model for Pipeline Parallelism.
Each PP rank owns exactly one stage containing a subset of transformer blocks.

Weight Tying:
    GPT-2 uses weight tying between input embeddings (wte) and output projection (lm_head).
    Since these are on different stages (first vs last), we copy wte to both stages
    and sync gradients after backward pass.

Stage Structure by PP rank:
┌─────────────────────────────────────────────────────────────────────────────┐
│  PP=2 Example (12 blocks total)                                             │
│                                                                             │
│  Stage 0 (pp_rank=0):                  Stage 1 (pp_rank=1):                 │
│  ├── embedding.wte [50257, 768]        ├── blocks[6..11]                    │
│  ├── embedding.wpe [1024, 768]         ├── ln_f                             │
│  ├── blocks[0..5]                      └── lm_head_weight [50257, 768]      │
│  └── (sends hidden to Stage 1)              ↑ COPY of wte (for weight tying)│
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
    - Final LayerNorm + lm_head (only last stage, pp_rank=pp_size-1)
    
    Weight Tying:
        wte is copied to both first stage (embedding) and last stage (lm_head).
        Gradients must be synced between stages after backward pass.
    """
    
    def __init__(
        self, 
        embedding: GPT2Embedding,
        blocks: nn.ModuleList,
        ln_f: nn.LayerNorm,
        lm_head_weight: nn.Parameter,
        is_first_stage: bool,
        is_last_stage: bool,
        pp_group: dist.ProcessGroup = None,
    ):
        super().__init__()
        
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage
        self.pp_group = pp_group
        
        # Conditionally set components
        self.embedding = embedding if is_first_stage else None
        self.blocks = blocks
        self.ln_f = ln_f if is_last_stage else None
        
        # ─────────────────────────────────────────────────────────────────
        # Weight Tying: lm_head uses a COPY of wte (on last stage only)
        # ─────────────────────────────────────────────────────────────────
        if is_last_stage and lm_head_weight is not None:
            self.lm_head_weight = lm_head_weight  # nn.Parameter
        else:
            self.lm_head_weight = None
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for this pipeline stage.
        
        Args:
            x: For first stage: input_ids [B, T]
               For other stages: hidden_states [B, T, embed_dim]
            position_ids: Position IDs [B, T] (only used by first stage)
        
        Returns:
            For last stage: logits [B, T, vocab_size]
            For other stages: hidden_states [B, T, embed_dim]
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
        # Last stage: Apply final layer norm and compute logits
        # ─────────────────────────────────────────────────────────────────
        if self.is_last_stage:
            x = self.ln_f(x)  # [B, T, 768]
            
            # Compute logits using lm_head (which is a copy of wte)
            # logits = hidden_states @ wte.T
            if self.lm_head_weight is not None:
                x = x @ self.lm_head_weight.T  # [B, T, 768] @ [768, 50257] → [B, T, 50257]
        
        return x
    
    def sync_tied_weights_grad(self):
        """
        Synchronize gradients for tied weights (wte / lm_head) between first and last stage.
        
        Call this AFTER loss.backward() and BEFORE optimizer.step().
        
        This averages the gradients from:
        - First stage: embedding.wte.weight.grad
        - Last stage: lm_head_weight.grad
        """
        if self.pp_group is None:
            return
        
        if self.is_first_stage and self.embedding is not None:
            # Average gradients with last stage
            if self.embedding.wte.weight.grad is not None:
                dist.all_reduce(
                    self.embedding.wte.weight.grad, 
                    op=dist.ReduceOp.AVG, 
                    group=self.pp_group
                )
        
        if self.is_last_stage and self.lm_head_weight is not None:
            # Average gradients with first stage
            if self.lm_head_weight.grad is not None:
                dist.all_reduce(
                    self.lm_head_weight.grad, 
                    op=dist.ReduceOp.AVG, 
                    group=self.pp_group
                )
    
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
        pp_group: dist.ProcessGroup,
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
            pp_group: Pipeline parallel process group (for weight sync)
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
        wte_weight = None
        
        if is_first_stage:
            wte_weight = state_dict['wte.weight']
            embedding = GPT2Embedding(
                vocab_size=config.vocab_size,
                max_position_embeddings=config.n_positions,
                embed_dim=embed_dim,
                wte_weights=wte_weight,
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
        # Build Final LayerNorm + lm_head (only last stage)
        # ═══════════════════════════════════════════════════════════════════
        ln_f = None
        lm_head_weight = None
        
        if is_last_stage:
            # Final LayerNorm
            ln_f = nn.LayerNorm(embed_dim, device=device)
            with torch.no_grad():
                ln_f.weight.copy_(state_dict['ln_f.weight'])
                ln_f.bias.copy_(state_dict['ln_f.bias'])
            
            # ─────────────────────────────────────────────────────────────────
            # Weight Tying: Copy wte to last stage for lm_head
            # Note: state_dict should have 'wte.weight' copied for last stage
            # ─────────────────────────────────────────────────────────────────
            if 'wte.weight' in state_dict:
                lm_head_weight = nn.Parameter(
                    state_dict['wte.weight'].clone().to(device)
                )
            else:
                # If wte not in state_dict for last stage, we need to handle this
                # This shouldn't happen if load_gpt2_distributed is correctly implemented
                raise ValueError(
                    "wte.weight not found in state_dict for last stage. "
                    "Ensure load_gpt2_distributed includes wte.weight for last stage (weight tying)."
                )
        
        # ═══════════════════════════════════════════════════════════════════
        # Create and return the stage instance
        # ═══════════════════════════════════════════════════════════════════
        return cls(
            embedding=embedding,
            blocks=blocks,
            ln_f=ln_f,
            lm_head_weight=lm_head_weight,
            is_first_stage=is_first_stage,
            is_last_stage=is_last_stage,
            pp_group=pp_group,
        )