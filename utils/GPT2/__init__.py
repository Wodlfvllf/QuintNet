"""
GPT-2 Model Components for Distributed Training

This module provides all the building blocks for GPT-2 with Tensor Parallelism (TP)
and Pipeline Parallelism (PP) support.

Exports:
    - GPT2Config: Configuration dataclass for model parameters
    - GPT2Embedding: Token + position embeddings
    - GPT2Attention: Multi-head attention with TP
    - GPT2MLP: Feed-forward network with TP
    - GPT2Block: Full transformer block
    - GPT2Stage: Pipeline parallelism stage container
"""

from .gpt2_config import GPT2Config
from .gpt2_block import GPT2Block
from .gpt2_embeddings import GPT2Embedding
from .gpt2_stage import GPT2Stage
from .gpt2_attention import GPT2Attention
from .gpt2_mlp import GPT2MLP

__all__ = [
    "GPT2Config",
    "GPT2Block",
    "GPT2Embedding",
    "GPT2Stage",
    "GPT2Attention",
    "GPT2MLP",
]
