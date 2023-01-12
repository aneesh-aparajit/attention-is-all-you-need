import torch as T 
import torch.nn as nn
from attention import MultiHeadAttention, scaled_dot_product
from typing import Optional

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, N: int = 6, num_heads: int = 8, mlp_dim: int = 768, p: float = 0.3) -> None:
        super().__init__()
        self.multihead = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), 
            nn.LayerNorm(normalized_shape=mlp_dim),
        )
        self.dropout = nn.Dropout(p=p)
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
    
    def forward(self, Q: T.Tensor, K: T.Tensor, V: T.Tensor, mask: Optional[T.Tensor] = None) -> T.Tensor:
        x, attention = self.multihead(Q, K, V, mask, return_attn=True)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

if __name__ == '__main__':
    model = TransformerBlock(embed_dim=512, num_heads=8)
    Q = T.randn(32, 128, 512)
    K = T.randn(32, 100, 512)
    V = T.randn(32, 164, 512)
    out = model.forward(Q, K, V)
    print(f'transformers: {out.shape}')
