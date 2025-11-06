import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Import utils from same package
from .utils import *

class Attention(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Define Q, K, V matrices here
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        B, Seq_len, _ = x.shape
        
        # Project inputs
        query = self.Q(x)
        key = self.K(x)
        value = self.V(x)
        
        # Split into heads
        Q_new = rearrange(query, 'b s (n h) -> b n s h', n=self.n_heads)
        K_new = rearrange(key, 'b s (n h) -> b n s h', n=self.n_heads)
        V_new = rearrange(value, 'b s (n h) -> b n s h', n=self.n_heads)
        
        # Compute attention scores
        attn = Q_new @ rearrange(K_new, 'b n s h -> b n h s')  # shape: (b, n, s, s)
        attn = attn / np.sqrt(self.head_dim)                   # scale by sqrt(d_k)
        
        # Softmax
        score = F.softmax(attn, dim=-1)                        # (b, n, s, s)
        
        # Weighted sum
        output = score @ V_new                                 # (b, n, s, h)
        
        # Merge heads
        out = rearrange(output, 'b n s h -> b s (n h)')
        
        return out

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.input_dim = input_dim
        
        self.down_proj = nn.Linear(input_dim,  64)
        self.relu = nn.ReLU()
        self.up_proj = nn.Linear(64,  input_dim)
        
    def forward(self, x):
        out = self.down_proj(x)
        out = self.up_proj(out)
        out = self.relu(out)
        
        return out
        
        return out
    
    
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Conv2d extracts patches & projects them
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # x shape: (batch, channels, H, W)
        x = self.proj(x) # (batch, embed_dim, H/patch, W/patch)
        x = x.flatten(2) # (batch, embed_dim, n_patches)
        x = x.transpose(1, 2) # (batch, n_patches, embed_dim)
        return x
        
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.attention = Attention(hidden_dim, n_heads)
        self.mlp = MLP(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # Note: ReLU is usually inside the MLP, not outside. I've left it for now.

    def forward(self, x):
        # Apply the first residual connection
        x = x + self.attention(self.norm1(x))
        # Apply the second residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes=10):
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is the full sequence from the last transformer block
        # Select the output of the [CLS] token, which is at index 0
        cls_out = x[:, 0]
        # Pass it through the final linear layer
        out = self.down_proj(cls_out)
        return out
    
# You can add this to your model.py file

class ViTEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, hidden_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.batch_size = None
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Calculate the number of patches correctly
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))

    def forward(self, x):
        # x: (B, C, H, W)
        out = self.patch_embed(x) # (B, N, D) where N is num_patches
        B, N, D = out.shape
        self.batch_size = B 

        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        out = torch.cat((cls_tokens, out), dim=1) # (B, N+1, D)
        
        # Add positional embedding
        out = out + self.pos_embed
        
        return out
    
class Model(nn.Module):
    def __init__(self, 
                img_size=28, 
                patch_size=4, 
                hidden_dim=64, 
                in_channels=1, 
                n_heads=4, 
                depth=4):
        super().__init__()
        
        # Use the new, encapsulated embedding module
        self.embedding = ViTEmbedding(img_size, patch_size, in_channels, hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, n_heads) for _ in range(depth)])
        self.classification_head = ClassificationHead(hidden_dim, num_classes=10)

    def forward(self, x):
        # The embedding logic is now cleanly handled in one module
        out = self.embedding(x)

        for blk in self.blocks:
            out = blk(out)

        # The classification head already correctly takes the CLS token
        # Your previous code had a bug here `self.down_proj`, this is the fix
        out = self.classification_head(out)
        return out


    
        
        
        