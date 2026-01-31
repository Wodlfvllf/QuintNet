

"""
GPT 2 Embeddings Implementation
"""

import torch
import torch.nn as nn
import torch.distributed as dist

class GPT2Embedding(nn.Module):
    def __init__(
        self, 
        token_embed_dim : int, 
        position_embed_dim : int, 
        wte_weights : torch.Tensor, 
        wpe_weights : torch.Tensor
        ):
    
        self.token_embed_dim = token_embed_dim
        self.position_embed_dim = position_embed_dim
        # Dim : ([50257, 768])
        self.wte_weights = wte_weights

        # Dim : ([1024, 768])
        self.wpe_weights = wpe_weights

        self.wte = nn.Linear(self.wte_weights.shape[0], self.wte_weights.shape[1])
        self.wpe = nn.Linear(self.wpe_weights.shape[0], self.wpe_weights.shape[1])

        with torch.no_grad():
            self.wte.weight.copy_(self.wte_weights)
            self.wpe.weight.copy_(self.wpe_weights)

    def forward(self, input_ids : torch.Tensor, position_ids : torch.Tensor) -> torch.Tensor:
        
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        return token_embeddings + position_embeddings
