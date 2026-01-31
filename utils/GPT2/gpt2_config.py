"""
GPT-2 Configuration Dataclass

Provides a structured configuration object for GPT-2 model parameters.
Enables attribute access (config.n_embd) instead of dict access (config['n_embd']).

Usage:
    # From dict (e.g., loaded from YAML):
    config = GPT2Config.from_dict({'n_embd': 768, 'n_head': 12, ...})
    
    # With defaults (GPT-2 base):
    config = GPT2Config()
    
    # Access parameters:
    print(config.n_embd)  # 768
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPT2Config:
    """
    Configuration for GPT-2 model architecture.
    
    Defaults correspond to GPT-2 base (124M parameters).
    
    Attributes:
        vocab_size: Size of the vocabulary (50257 for GPT-2)
        n_positions: Maximum sequence length (1024 for GPT-2)
        n_embd: Embedding dimension (768 for base, 1024 for medium, 1280 for large)
        n_layer: Number of transformer layers (12 for base)
        n_head: Number of attention heads (12 for base)
        n_inner: Intermediate MLP dimension (default: 4 * n_embd)
        attn_pdrop: Attention dropout probability
        embd_pdrop: Embedding dropout probability
        resid_pdrop: Residual dropout probability
        layer_norm_epsilon: Epsilon for LayerNorm stability
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID (GPT-2 uses eos for padding)
    """
    
    # Core architecture
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: Optional[int] = None  # Default: 4 * n_embd
    
    # Dropout rates
    attn_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    
    # LayerNorm
    layer_norm_epsilon: float = 1e-5
    
    # Special tokens
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    pad_token_id: int = 50256  # GPT-2 uses eos as pad token
    
    def __post_init__(self):
        """Set n_inner to 4 * n_embd if not specified."""
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "GPT2Config":
        """
        Create GPT2Config from a dictionary.
        
        Only uses keys that match GPT2Config fields, ignores others.
        This allows passing a full config dict that may have extra keys
        like 'batch_size', 'learning_rate', etc.
        
        Args:
            config_dict: Dictionary with config values
        
        Returns:
            GPT2Config instance
        
        Example:
            >>> config = GPT2Config.from_dict({
            ...     'n_embd': 1024,
            ...     'n_layer': 24,
            ...     'batch_size': 8,  # Ignored - not a GPT2Config field
            ... })
        """
        # Get the field names of the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        
        # Filter to only include valid fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_model_config(cls, model_config: dict) -> "GPT2Config":
        """
        Create GPT2Config from a 'model_config' sub-dict.
        
        This is useful when your YAML config has:
            model_config:
              n_embd: 768
              n_layer: 12
              ...
        
        Args:
            model_config: The 'model_config' sub-dictionary
        
        Returns:
            GPT2Config instance
        """
        return cls.from_dict(model_config)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'n_positions': self.n_positions,
            'n_embd': self.n_embd,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_inner': self.n_inner,
            'attn_pdrop': self.attn_pdrop,
            'embd_pdrop': self.embd_pdrop,
            'resid_pdrop': self.resid_pdrop,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
        }
    
    @classmethod
    def gpt2_base(cls) -> "GPT2Config":
        """GPT-2 Base (124M parameters)."""
        return cls()
    
    @classmethod
    def gpt2_medium(cls) -> "GPT2Config":
        """GPT-2 Medium (355M parameters)."""
        return cls(
            n_embd=1024,
            n_layer=24,
            n_head=16,
        )
    
    @classmethod
    def gpt2_large(cls) -> "GPT2Config":
        """GPT-2 Large (774M parameters)."""
        return cls(
            n_embd=1280,
            n_layer=36,
            n_head=20,
        )
    
    @classmethod
    def gpt2_xl(cls) -> "GPT2Config":
        """GPT-2 XL (1.5B parameters)."""
        return cls(
            n_embd=1600,
            n_layer=48,
            n_head=25,
        )
