

"""
The Problem: Why Distributed Loading?
Current Approach:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT FLOW                                      │
│                                                                             │
│   GPU 0        GPU 1        GPU 2        GPU 3        GPU 4   ...  GPU 7    │
│   ┌────┐       ┌────┐       ┌────┐       ┌────┐       ┌────┐       ┌────┐   │
│   │FULL│       │FULL│       │FULL│       │FULL│       │FULL│       │FULL│   │
│   │ 500│       │ 500│       │ 500│       │ 500│       │ 500│       │ 500│   │
│   │ MB │       │ MB │       │ MB │       │ MB │       │ MB │       │ MB │   │
│   └────┘       └────┘       └────┘       └────┘       └────┘       └────┘   │
│      │            │            │            │            │            │     │
│      └────────────┴────────────┴─────┬──────┴────────────┴────────────┘     │
│                                      │                                      │
│                                      ▼                                      │
│                          ┌─────────────────────┐                            │
│                          │  Apply TP/PP/DP     │                            │
│                          │  (slice & discard)  │                            │
│                          └─────────────────────┘                            │
│                                      │                                      │
│                                      ▼                                      │
│   GPU 0        GPU 1        GPU 2        GPU 3        GPU 4   ...  GPU 7    │
│   ┌────┐       ┌────┐       ┌────┐       ┌────┐       ┌────┐       ┌────┐   │
│   │ 62 │       │ 62 │       │ 62 │       │ 62 │       │ 62 │       │ 62 │   │
│   │ MB │       │ MB │       │ MB │       │ MB │       │ MB │       │ MB │   │
│   └────┘       └────┘       └────┘       └────┘       └────┘       └────┘   │
│                                                                             │
│   Peak Memory Per GPU: 500MB  →  Only 62MB used!                            │
│   Wasted: 438MB per GPU (87.6% waste!)                                      │
└─────────────────────────────────────────────────────────────────────────────┘

The Problem with Scaling
Model	            Size	        8 GPUs Current	    8 GPUs Distributed
GPT-2	            500MB	        4GB peak	        500MB peak
GPT-2 XL	        6GB	            48GB peak	        6GB peak
LLaMA-7B	        14GB	        OOM!	            14GB peak
LLaMA-70B	        140GB	        OOM!	            140GB peak



GPT-2 Model Structure
═══════════════════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────────────────┐
│  EMBEDDINGS (replicated on all TP ranks)                            │
│  ├─ wte.weight: [50257, 768]  Token Embeddings                      │
│  └─ wpe.weight: [1024, 768]   Position Embeddings                   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TRANSFORMER BLOCK × 12 (h.0 to h.11)                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  ┌─────────────┐                                              │  │
│  │  │ LayerNorm 1 │  ln_1.weight/bias: [768] (replicated)        │  │
│  │  └──────┬──────┘                                              │  │
│  │         │                                                     │  │
│  │         ▼                                                     │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │                 ATTENTION BLOCK                         │  │  │
│  │  │                                                         │  │  │
│  │  │   Input [B, S, 768]                                     │  │  │
│  │  │         │                                               │  │  │
│  │  │         ▼                                               │  │  │
│  │  │   ┌──────────────┐                                      │  │  │
│  │  │   │   c_attn     │  W: [768, 2304]  ← COLUMN PARALLEL   │  │  │
│  │  │   │   (Q,K,V)    │  b: [2304]         (shard by 3×heads)│  │  │
│  │  │   └──────────────┘                                      │  │  │
│  │  │         │                                               │  │  │
│  │  │         ▼  [B, S, 2304] → split → Q,K,V each [B,S,768]  │  │  │
│  │  │   ┌──────────────┐                                      │  │  │
│  │  │   │   Attention  │  (local per TP rank)                 │  │  │
│  │  │   └──────────────┘                                      │  │  │
│  │  │         │                                               │  │  │
│  │  │         ▼  [B, S, 768/TP_SIZE] (partial)                │  │  │
│  │  │   ┌──────────────┐                                      │  │  │
│  │  │   │   c_proj     │  W: [768, 768]  ← ROW PARALLEL       │  │  │
│  │  │   │   (output)   │  b: [768]         (shard by input)   │  │  │
│  │  │   └──────────────┘                                      │  │  │
│  │  │         │                                               │  │  │
│  │  │         ▼  AllReduce → [B, S, 768] (full)               │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │         │                                                     │  │
│  │         ▼  + Residual                                         │  │
│  │  ┌─────────────┐                                              │  │
│  │  │ LayerNorm 2 │  ln_2.weight/bias: [768] (replicated)        │  │
│  │  └──────┬──────┘                                              │  │
│  │         │                                                     │  │
│  │         ▼                                                     │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │                     MLP BLOCK                           │  │  │
│  │  │                                                         │  │  │
│  │  │   Input [B, S, 768]                                     │  │  │
│  │  │         │                                               │  │  │
│  │  │         ▼                                               │  │  │
│  │  │   ┌──────────────┐                                      │  │  │
│  │  │   │   c_fc       │  W: [768, 3072]  ← COLUMN PARALLEL   │  │  │
│  │  │   │   (up proj)  │  b: [3072]         (expand hidden)   │  │  │
│  │  │   └──────────────┘                                      │  │  │
│  │  │         │                                               │  │  │
│  │  │         ▼  [B, S, 3072/TP_SIZE] (partial)               │  │  │
│  │  │   ┌──────────────┐                                      │  │  │
│  │  │   │     GELU     │  (local, no communication)           │  │  │
│  │  │   └──────────────┘                                      │  │  │
│  │  │         │                                               │  │  │
│  │  │         ▼                                               │  │  │
│  │  │   ┌──────────────┐                                      │  │  │
│  │  │   │   c_proj     │  W: [3072, 768]  ← ROW PARALLEL      │  │  │
│  │  │   │   (down proj)│  b: [768]          (reduce hidden)   │  │  │
│  │  │   └──────────────┘                                      │  │  │
│  │  │         │                                               │  │  │
│  │  │         ▼  AllReduce → [B, S, 768] (full)               │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │         │                                                     │  │
│  │         ▼  + Residual → Output [B, S, 768]                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼ (repeat × 12)
┌─────────────────────────────────────────────────────────────────────┐
│  FINAL LAYER NORM                                                   │
│  └─ ln_f.weight/bias: [768] (replicated)                            │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LM HEAD                                                            │
│  └─ lm_head.weight: [768, 50257] (often tied to wte.weight)         │
└─────────────────────────────────────────────────────────────────────┘


3D Device Mesh Visualization
═══════════════════════════════════════════════════════════════════════
Physical Layout:  8 GPUs numbered 0-7
Logical 3D Mesh:  mesh_dim = [2, 2, 2]  (DP × TP × PP)
                  mesh_name = ['dp', 'tp', 'pp']
                              PP Dimension
                           ┌─────────────────┐
                          ╱│                ╱│
                         ╱ │    Stage 1    ╱ │
                        ╱  │              ╱  │
                       ┌───┼─────────────┐   │
            TP         │   │             │   │
         Dimension     │   │DP Replica 1 │   │
              ▲        │   └─────────────┼───┘
              │        │  ╱              │  ╱
              │        │ ╱   Stage 0     │ ╱
              │        │╱                │╱
              │        └─────────────────┘
              │            DP Replica 0
              │
              └──────────────────────────────→ DP Dimension


Rank to Coordinate Mapping:
═══════════════════════════
Global   (DP, TP, PP)   Role
Rank     Coordinates
──────────────────────────────────────────────────────────────────────
  0      (0,  0,  0)    DP Replica 0, TP Shard 0, PP Stage 0 (first)
  1      (0,  0,  1)    DP Replica 0, TP Shard 0, PP Stage 1 (last)
  2      (0,  1,  0)    DP Replica 0, TP Shard 1, PP Stage 0 (first)
  3      (0,  1,  1)    DP Replica 0, TP Shard 1, PP Stage 1 (last)
  4      (1,  0,  0)    DP Replica 1, TP Shard 0, PP Stage 0 (first)
  5      (1,  0,  1)    DP Replica 1, TP Shard 0, PP Stage 1 (last)
  6      (1,  1,  0)    DP Replica 1, TP Shard 1, PP Stage 0 (first)
  7      (1,  1,  1)    DP Replica 1, TP Shard 1, PP Stage 1 (last)


Process Groups (Who Communicates With Whom):
════════════════════════════════════════════
DP Groups (gradient sync - AllReduce):
  • [0, 4]  - Same TP shard 0, same PP stage 0
  • [1, 5]  - Same TP shard 0, same PP stage 1
  • [2, 6]  - Same TP shard 1, same PP stage 0
  • [3, 7]  - Same TP shard 1, same PP stage 1
TP Groups (tensor parallel - AllGather/AllReduce):
  • [0, 2]  - Same DP replica 0, same PP stage 0
  • [1, 3]  - Same DP replica 0, same PP stage 1
  • [4, 6]  - Same DP replica 1, same PP stage 0
  • [5, 7]  - Same DP replica 1, same PP stage 1
PP Groups (pipeline parallel - P2P Send/Recv):
  • [0, 1]  - Same DP replica 0, same TP shard 0
  • [2, 3]  - Same DP replica 0, same TP shard 1
  • [4, 5]  - Same DP replica 1, same TP shard 0
  • [6, 7]  - Same DP replica 1, same TP shard 1



"""


import torch
import torch.nn as nn
import torch.distributed as dist
from .process_groups import ProcessGroupManager, init_process_groups
from .mesh import MeshGenerator
from safetensors import safe_open

def load_gpt2_distributed(checkpoint_path, pg_manager, config, device, model_config=None):
    """
    Load GPT-2 checkpoint in a distributed manner across the 3D mesh.
    
    Args:
        checkpoint_path: Path to safetensors checkpoint
        pg_manager: ProcessGroupManager for mesh coordinates
        config: Training config dict (for mesh_dim, mesh_name)
        device: Device for this GPU
        model_config: GPT2Config object with model parameters (n_layer, n_embd, etc.)
                     If None, uses default GPT-2 base settings.
    """

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Determine this GPU's role in the mesh
    # ═══════════════════════════════════════════════════════════════════
    
    global_rank = dist.get_rank()
    coords = pg_manager.get_coordinates_tensor_search(global_rank)

    dp_rank = coords[config['mesh_name'].index('dp')]  # Which DP replica
    tp_rank = coords[config['mesh_name'].index('tp')]  # Which TP shard
    pp_rank = coords[config['mesh_name'].index('pp')]  # Which PP stage
    
    dp_size = config['mesh_dim'][config['mesh_name'].index('dp')]
    tp_size = config['mesh_dim'][config['mesh_name'].index('tp')]
    pp_size = config['mesh_dim'][config['mesh_name'].index('pp')]

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Calculate which layers this GPU owns (Pipeline Parallelism)
    # ═══════════════════════════════════════════════════════════════════

    # Get model dimensions from model_config (or use defaults)
    if model_config is not None:
        num_layers = model_config.n_layer
        embed_dim = model_config.n_embd
        hidden_dim = model_config.n_inner or 4 * embed_dim
    else:
        # Default GPT-2 base dimensions
        num_layers = 12
        embed_dim = 768
        hidden_dim = 3072
    
    layers_per_stage = num_layers // pp_size

    my_layer_start = pp_rank * layers_per_stage
    my_layer_end = my_layer_start + layers_per_stage
    my_layers = list(range(my_layer_start, my_layer_end))

    # Example with PP=2:
    #   pp_rank=0: layers [0, 1, 2, 3, 4, 5]
    #   pp_rank=1: layers [6, 7, 8, 9, 10, 11]

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Open safetensors file (memory-mapped, NOT loaded to RAM!)
    # ═══════════════════════════════════════════════════════════════════
    
    state_dict = {}

    with safe_open(checkpoint_path, framework='pt', device=str(device)) as f:
        # ═══════════════════════════════════════════════════════════════
        # STEP 4A: Load embeddings (only first PP stage needs them)
        # ═══════════════════════════════════════════════════════════════

        if pp_rank == 0: # First stage
            # Embeddings are replicated across TP Ranks and Not sharded. 
            # Vocab Embeddings
            state_dict['wte.weight'] = f.get_tensor('wte.weight') # [50257, 768]

            # Positional Encodings
            state_dict['wpe.weight'] = f.get_tensor('wpe.weight') # [1024, 768]

        # ═══════════════════════════════════════════════════════════════
        # STEP 4B: Load transformer blocks (only my PP stage's layers)
        # ═══════════════════════════════════════════════════════════════

        for layer_idx in my_layers:
            prefix = f"h.{layer_idx}"

            # ─────────────────────────────────────────────────────────
            # LayerNorms: REPLICATED (each TP rank has full copy)
            # ─────────────────────────────────────────────────────────

            state_dict[f"{prefix}.ln_1.weight"] = f.get_tensor(f"{prefix}.ln_1.weight")
            state_dict[f"{prefix}.ln_1.bias"] = f.get_tensor(f"{prefix}.ln_1.bias")
            state_dict[f"{prefix}.ln_2.weight"] = f.get_tensor(f"{prefix}.ln_2.weight")
            state_dict[f"{prefix}.ln_2.bias"] = f.get_tensor(f"{prefix}.ln_2.bias")

            # ─────────────────────────────────────────────────────────
            # Attention c_attn: COLUMN PARALLEL (shard output dim)
            #   Full: [embed_dim, 3*embed_dim] → Each TP rank: [embed_dim, 3*embed_dim/tp_size]
            # ─────────────────────────────────────────────────────────
            c_attn_w_full = f.get_slice(f"{prefix}.attn.c_attn.weight")
            c_attn_b_full = f.get_slice(f"{prefix}.attn.c_attn.bias")
            
            c_attn_out_dim = 3 * embed_dim  # 3 * 768 = 2304 for GPT-2 base
            cols_per_rank = c_attn_out_dim // tp_size
            col_start = tp_rank * cols_per_rank
            col_end = col_start + cols_per_rank
            
            # The magic: read ONLY the slice we need!
            state_dict[f"{prefix}.attn.c_attn.weight"] = c_attn_w_full[:, col_start:col_end]
            state_dict[f"{prefix}.attn.c_attn.bias"] = c_attn_b_full[col_start:col_end]


            # ─────────────────────────────────────────────────────────
            # Attention c_proj: ROW PARALLEL (shard input dim)
            #   Full: [embed_dim, embed_dim] → Each TP rank: [embed_dim/tp_size, embed_dim]
            # ─────────────────────────────────────────────────────────
            c_proj_w_full = f.get_slice(f"{prefix}.attn.c_proj.weight")
            
            rows_per_rank = embed_dim // tp_size
            row_start = tp_rank * rows_per_rank
            row_end = row_start + rows_per_rank
            
            state_dict[f"{prefix}.attn.c_proj.weight"] = c_proj_w_full[row_start:row_end, :]
            
            # Bias: only tp_rank=0 stores it (to avoid double-add after AllReduce)
            if tp_rank == 0:
                state_dict[f"{prefix}.attn.c_proj.bias"] = f.get_tensor(f"{prefix}.attn.c_proj.bias")

            # ─────────────────────────────────────────────────────────
            # MLP c_fc: COLUMN PARALLEL (shard output dim)
            #   Full: [embed_dim, hidden_dim] → Each TP rank: [embed_dim, hidden_dim/tp_size]
            # ─────────────────────────────────────────────────────────
            c_fc_w_full = f.get_slice(f"{prefix}.mlp.c_fc.weight")
            c_fc_b_full = f.get_slice(f"{prefix}.mlp.c_fc.bias")
            
            cols_per_rank = hidden_dim // tp_size
            col_start = tp_rank * cols_per_rank
            col_end = col_start + cols_per_rank
            
            state_dict[f"{prefix}.mlp.c_fc.weight"] = c_fc_w_full[:, col_start:col_end]
            state_dict[f"{prefix}.mlp.c_fc.bias"] = c_fc_b_full[col_start:col_end]

            # ─────────────────────────────────────────────────────────
            # MLP c_proj: ROW PARALLEL (shard input dim)
            #   Full: [hidden_dim, embed_dim] → Each TP rank: [hidden_dim/tp_size, embed_dim]
            # ─────────────────────────────────────────────────────────
            c_proj_w_full = f.get_slice(f"{prefix}.mlp.c_proj.weight")
            
            rows_per_rank = hidden_dim // tp_size
            row_start = tp_rank * rows_per_rank
            row_end = row_start + rows_per_rank
            
            state_dict[f"{prefix}.mlp.c_proj.weight"] = c_proj_w_full[row_start:row_end, :]
            
            if tp_rank == 0:
                state_dict[f"{prefix}.mlp.c_proj.bias"] = f.get_tensor(f"{prefix}.mlp.c_proj.bias")

            
        # ═══════════════════════════════════════════════════════════════
        # STEP 4C: Load final layer norm (only last PP stage needs it)
        # ═══════════════════════════════════════════════════════════════
        
        if pp_rank == pp_size - 1:  # Last stage
            state_dict["ln_f.weight"] = f.get_tensor("ln_f.weight")
            state_dict["ln_f.bias"] = f.get_tensor("ln_f.bias")
            
            # ─────────────────────────────────────────────────────────────
            # Weight Tying: Load wte.weight for lm_head (copy of embedding)
            # GPT-2 uses weight tying: lm_head.weight = wte.weight.T
            # Since embedding is on first stage, we need wte on last stage too
            # ─────────────────────────────────────────────────────────────
            state_dict['wte.weight'] = f.get_tensor('wte.weight')  # [50257, 768]
    
    return state_dict




    