

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
from .process_groups import ProcessGroupManager, init_process_groups
from .mesh import MeshGenerator

