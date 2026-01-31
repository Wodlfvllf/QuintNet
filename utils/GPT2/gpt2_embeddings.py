"""
GPT-2 Embeddings Implementation

Embeddings in GPT-2:
- wte (Word Token Embeddings): Maps token IDs to vectors [vocab_size, embed_dim]
- wpe (Word Position Embeddings): Maps position IDs to vectors [max_seq_len, embed_dim]

Note: Embeddings are REPLICATED across all TP ranks (not sharded).
Each GPU needs the full embedding table to look up any token.
"""

import torch
import torch.nn as nn


class GPT2Embedding(nn.Module):
    """
    GPT-2 Embedding layer combining token and position embeddings.
    
    Args:
        vocab_size: Size of vocabulary (50257 for GPT-2)
        max_position_embeddings: Maximum sequence length (1024 for GPT-2)
        embed_dim: Embedding dimension (768 for GPT-2)
        wte_weights: Pre-trained token embedding weights [vocab_size, embed_dim]
        wpe_weights: Pre-trained position embedding weights [max_seq_len, embed_dim]
        dropout_prob: Dropout probability for embeddings
        device: Device to place embeddings on
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        embed_dim: int,
        wte_weights: torch.Tensor,
        wpe_weights: torch.Tensor,
        dropout_prob: float = 0.1,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.embed_dim = embed_dim
        
        # ═══════════════════════════════════════════════════════════════════
        # Token Embeddings (wte): [50257, 768]
        # Lookup table: token_id → embedding vector
        # ═══════════════════════════════════════════════════════════════════
        self.wte = nn.Embedding(vocab_size, embed_dim, device=device)
        
        # ═══════════════════════════════════════════════════════════════════
        # Position Embeddings (wpe): [1024, 768]
        # Lookup table: position → position embedding vector
        # ═══════════════════════════════════════════════════════════════════
        self.wpe = nn.Embedding(max_position_embeddings, embed_dim, device=device)
        
        # Dropout applied after combining embeddings
        self.dropout = nn.Dropout(dropout_prob)
        
        # ═══════════════════════════════════════════════════════════════════
        # Load pre-trained weights
        # ═══════════════════════════════════════════════════════════════════
        with torch.no_grad():
            self.wte.weight.copy_(wte_weights)
            self.wpe.weight.copy_(wpe_weights)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        position_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len] or None
                         If None, automatically generated as [0, 1, 2, ..., seq_len-1]
        
        Returns:
            embeddings: Combined embeddings [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Look up token embeddings: [B, T] → [B, T, 768]
        token_embeddings = self.wte(input_ids)
        
        # Look up position embeddings: [B, T] → [B, T, 768]
        position_embeddings = self.wpe(position_ids)
        
        # Combine: token + position
        embeddings = token_embeddings + position_embeddings
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
