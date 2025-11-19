"""
Vision Transformer (ViT) Model Architecture

This module defines the core components and the complete Vision Transformer (ViT)
model architecture used in QuintNet. It includes implementations of attention
mechanisms, Multi-Layer Perceptrons (MLPs), patch embeddings, transformer blocks,
and the overall ViT model structure.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

The Vision Transformer (ViT) adapts the Transformer architecture, originally
designed for natural language processing, to image classification tasks.
It treats images as sequences of "patches," similar to how words form a sentence.

Key components:
-   **`PatchEmbedding`**: Converts image patches into a sequence of embeddings.
-   **`ViTEmbedding`**: Combines patch embeddings with a learnable class token
    (`[CLS]`) and positional embeddings.
-   **`Attention`**: Implements multi-head self-attention, allowing the model
    to weigh the importance of different patches relative to each other.
-   **`MLP`**: A simple feed-forward network applied after attention.
-   **`TransformerBlock`**: Stacks attention and MLP layers with residual
    connections and layer normalization.
-   **`ClassificationHead`**: A linear layer that takes the `[CLS]` token's
    output and maps it to class probabilities.
-   **`Model`**: The complete ViT model, orchestrating all these components.

This modular design allows for easy modification and extension of the ViT
architecture.

===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np # Used for np.sqrt

# Import utils from same package (assuming utils.py contains necessary helper functions)
# from QuintNet.utils.utils import * # Specific imports are better than wildcard

class Attention(nn.Module):
    """
    Implements a Multi-Head Self-Attention mechanism.

    This layer allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    def __init__(self, hidden_dim: int, n_heads: int):
        """
        Initializes the Attention module.

        Args:
            hidden_dim (int): The dimension of the input and output features.
            n_heads (int): The number of attention heads. `hidden_dim` must be
                divisible by `n_heads`.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")

        # Linear layers for Query, Key, and Value projections
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, sequence_length, hidden_dim)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, sequence_length, hidden_dim)`.
        """
        B, Seq_len, _ = x.shape
        
        # Project inputs to Query, Key, Value spaces
        query = self.Q(x) # (B, Seq_len, hidden_dim)
        key = self.K(x)   # (B, Seq_len, hidden_dim)
        value = self.V(x) # (B, Seq_len, hidden_dim)
        
        # Reshape for multi-head attention: split hidden_dim into n_heads * head_dim
        Q_new = rearrange(query, 'b s (n h) -> b n s h', n=self.n_heads) # (B, n_heads, Seq_len, head_dim)
        K_new = rearrange(key, 'b s (n h) -> b n s h', n=self.n_heads)   # (B, n_heads, Seq_len, head_dim)
        V_new = rearrange(value, 'b s (n h) -> b n s h', n=self.n_heads) # (B, n_heads, Seq_len, head_dim)
        
        # Compute attention scores: (Q @ K^T) / sqrt(head_dim)
        # Matmul between Q and K_new.transpose(-2, -1)
        attn = Q_new @ rearrange(K_new, 'b n s h -> b n h s')  # shape: (B, n_heads, Seq_len, Seq_len)
        attn = attn / np.sqrt(self.head_dim)                   # Scale by sqrt(d_k)
        
        # Apply softmax to get attention probabilities
        score = F.softmax(attn, dim=-1)                        # (B, n_heads, Seq_len, Seq_len)
        
        # Apply attention scores to Value: (score @ V)
        output = score @ V_new                                 # (B, n_heads, Seq_len, head_dim)
        
        # Concatenate heads back together
        out = rearrange(output, 'b n s h -> b s (n h)')        # (B, Seq_len, hidden_dim)
        
        return out

class MLP(nn.Module):
    """
    Implements a simple Multi-Layer Perceptron (MLP) with a ReLU activation.
    This is typically used as the feed-forward component within a Transformer block.
    """
    def __init__(self, input_dim: int):
        """
        Initializes the MLP module.

        Args:
            input_dim (int): The dimension of the input and output features.
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Down-projection layer
        self.down_proj = nn.Linear(input_dim,  input_dim * 4) # Common practice to expand hidden dim
        self.relu = nn.ReLU()
        # Up-projection layer
        self.up_proj = nn.Linear(input_dim * 4,  input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, sequence_length, input_dim)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, sequence_length, input_dim)`.
        """
        out = self.down_proj(x)
        out = self.relu(out) # Activation after down-projection
        out = self.up_proj(out)
        
        return out
        
class PatchEmbedding(nn.Module):
    """
    Converts an input image into a sequence of flattened, linearly projected patches.
    """
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 64):
        """
        Initializes the PatchEmbedding module.

        Args:
            img_size (int): The height and width of the input image (assumed square).
            patch_size (int): The size of each square patch.
            in_channels (int): The number of input channels in the image (e.g., 3 for RGB, 1 for grayscale).
            embed_dim (int): The dimension of the output embedding for each patch.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Calculate the number of patches along one dimension
        num_patches_side = img_size // patch_size
        self.n_patches = num_patches_side * num_patches_side
        
        # Use a Conv2d layer to extract patches and project them to embed_dim.
        # kernel_size=patch_size and stride=patch_size ensure non-overlapping patches.
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the PatchEmbedding module.

        Args:
            x (torch.Tensor): Input image tensor of shape `(batch_size, in_channels, img_size, img_size)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, n_patches, embed_dim)`.
        """
        # x shape: (batch, channels, H, W)
        x = self.proj(x) # (batch, embed_dim, H/patch, W/patch)
        x = x.flatten(2) # (batch, embed_dim, n_patches) - flatten spatial dimensions
        x = x.transpose(1, 2) # (batch, n_patches, embed_dim) - transpose to sequence-first format
        return x
        
class TransformerBlock(nn.Module):
    """
    Implements a single Transformer block, consisting of Multi-Head Self-Attention
    and a Multi-Layer Perceptron, each with residual connections and Layer Normalization.
    """
    def __init__(self, hidden_dim: int, n_heads: int):
        """
        Initializes the TransformerBlock.

        Args:
            hidden_dim (int): The dimension of the input and output features.
            n_heads (int): The number of attention heads.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = Attention(hidden_dim, n_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, sequence_length, hidden_dim)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, sequence_length, hidden_dim)`.
        """
        # First sub-layer: Multi-Head Self-Attention with pre-normalization and residual connection
        x = x + self.attention(self.norm1(x))
        # Second sub-layer: MLP with pre-normalization and residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class ClassificationHead(nn.Module):
    """
    The final classification head of the Vision Transformer.

    It takes the output corresponding to the `[CLS]` token from the Transformer
    encoder and projects it to the number of output classes.
    """
    def __init__(self, hidden_dim: int, num_classes: int = 10):
        """
        Initializes the ClassificationHead.

        Args:
            hidden_dim (int): The dimension of the input feature vector (from the CLS token).
            num_classes (int): The number of output classes for classification.
        """
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the ClassificationHead.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, sequence_length, hidden_dim)`.
                This is the output from the last Transformer block.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, num_classes)`,
                representing the logits for each class.
        """
        # Extract the output corresponding to the [CLS] token (first token in the sequence)
        cls_out = x[:, 0] # (batch_size, hidden_dim)
        # Pass it through the final linear layer to get class logits
        out = self.down_proj(cls_out)
        return out
    
class ViTEmbedding(nn.Module):
    """
    Combines patch embeddings, a learnable class token, and positional embeddings
    to form the input sequence for the Transformer encoder.
    """
    def __init__(self, img_size: int = 28, patch_size: int = 4, in_channels: int = 1, hidden_dim: int = 64):
        """
        Initializes the ViTEmbedding module.

        Args:
            img_size (int): The height and width of the input image.
            patch_size (int): The size of each square patch.
            in_channels (int): The number of input channels in the image.
            hidden_dim (int): The dimension of the embeddings.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Patch embedding layer
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, hidden_dim)
        
        # Learnable [CLS] token, prepended to the sequence of patch embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Positional embeddings, added to patch and CLS embeddings
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the ViTEmbedding module.

        Args:
            x (torch.Tensor): Input image tensor of shape `(batch_size, in_channels, img_size, img_size)`.

        Returns:
            torch.Tensor: Output sequence tensor of shape `(batch_size, num_patches + 1, hidden_dim)`.
        """
        # Get patch embeddings
        out = self.patch_embed(x) # (B, N, D) where N is num_patches
        B, N, D = out.shape
        
        # Expand CLS token to match batch size and concatenate
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        out = torch.cat((cls_tokens, out), dim=1) # (B, N+1, D)
        
        # Add positional embedding
        out = out + self.pos_embed
        
        return out
    
class Model(nn.Module):
    """
    The complete Vision Transformer (ViT) model for image classification.
    """
    def __init__(self, 
                img_size: int = 28, 
                patch_size: int = 4, 
                hidden_dim: int = 64, 
                in_channels: int = 1, 
                n_heads: int = 4, 
                depth: int = 4,
                num_classes: int = 10):
        """
        Initializes the ViT Model.

        Args:
            img_size (int): The height and width of the input image.
            patch_size (int): The size of each square patch.
            hidden_dim (int): The dimension of the embeddings and transformer blocks.
            in_channels (int): The number of input channels in the image.
            n_heads (int): The number of attention heads in each Transformer block.
            depth (int): The number of Transformer blocks in the encoder.
            num_classes (int): The number of output classes for classification.
        """
        super().__init__()
        
        # Embedding layer (patch + CLS token + positional)
        self.embedding = ViTEmbedding(img_size, patch_size, in_channels, hidden_dim)
        
        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, n_heads) for _ in range(depth)])
        
        # Final classification head
        self.classification_head = ClassificationHead(hidden_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the complete ViT model.

        Args:
            x (torch.Tensor): Input image tensor of shape `(batch_size, in_channels, img_size, img_size)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, num_classes)`,
                representing the logits for each class.
        """
        # Process input through the embedding layer
        out = self.embedding(x)

        # Pass the sequence through all Transformer blocks
        for blk in self.blocks:
            out = blk(out)

        # Pass the output (specifically the CLS token) through the classification head
        out = self.classification_head(out)
        return out

    def get_tensor_shapes(self, batch_size: int) -> tuple:
        """
        A helper function to determine the shape of tensors that would be
        communicated between pipeline stages if this model were pipeline parallel.

        This is primarily used by the `PipelineParallelWrapper` to set up
        communication buffers.

        Args:
            batch_size (int): The batch size.

        Returns:
            tuple: The shape of the tensor that would be passed between stages.
        """
        # The shape of the tensor after embedding and before the first transformer block
        # is (batch_size, num_patches + 1, hidden_dim).
        num_patches = (self.embedding.img_size // self.embedding.patch_size)**2
        return (batch_size, num_patches + 1, self.embedding.hidden_dim)