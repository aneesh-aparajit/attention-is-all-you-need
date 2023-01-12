import torch as T
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple 
import math


def scaled_dot_product(Q: T.Tensor, K: T.Tensor, V: T.Tensor, mask: Optional[T.Tensor] = None) -> Tuple[T.Tensor, T.Tensor]:
    d_k = Q.shape[-1] # embedding dimension
    # print(Q.shape, K.shape)
    attn_logits = T.einsum("nqhd,nkhd->nhqk", Q, K) / math.sqrt(d_k)

    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask, float("-1e19"))
    
    attention = F.softmax(attn_logits, dim=-1) # dim = -1 indicates the horizontal axis

    print(V.shape, attention.shape)
    values = T.einsum("nhqk,nvhd->nvhd", attention, V)

    return values, attention


class MultiHeadAttention(nn.Module):
    '''
    The scaled dot product attention allows a network to attend over a sequence. However, often there are multiple different aspects of a sequence element wants to attend to, and a single weighted average is not a good option for it. This is why, we extend the attention mechanisms to multiple heads, i.e. multiple different query-key-value triplets on the same features. Specifically, given a query, key and value matrix, we transform those to "h" sub-queries, sub-keys, and sub-valyes, which we pass through a scaled dot product attention independantly. Afterward, we concatenate the heads and combine them with a final weight matrix.
    '''
    def __init__(self, embed_dim: int, num_heads: int, d_model: int = 512) -> None:
        super().__init__()

        assert embed_dim % num_heads == 0, "Embedding dim should be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.heads_dim = embed_dim // num_heads 
        self.d_model = d_model

        self.Q_proj = nn.Linear(self.heads_dim, self.heads_dim)
        self.K_proj = nn.Linear(self.heads_dim, self.heads_dim)
        self.V_proj = nn.Linear(self.heads_dim, self.heads_dim)
        self.O_proj = nn.Linear(self.num_heads*self.heads_dim, self.d_model)


    def forward(self, Q: T.Tensor, K: T.Tensor, V: T.Tensor, mask: Optional[T.Tensor] = None) -> T.Tensor:
        N = Q.shape[0] # batch size
        Q_len, K_len, V_len = Q.shape[1], K.shape[1], V.shape[1] # len of the sequence

        Q = Q.reshape(N, Q_len, self.num_heads, self.heads_dim)
        K = K.reshape(N, K_len, self.num_heads, self.heads_dim)
        V = V.reshape(N, V_len, self.num_heads, self.heads_dim)

        print(f'Q: {Q.shape}; K: {K.shape}; V: {V.shape}')

        Q = self.Q_proj(Q)
        V = self.V_proj(V)
        K = self.K_proj(K)

        print(f'Q: {Q.shape}; K: {K.shape}; V: {V.shape}')

        values, attention = scaled_dot_product(Q, K, V, mask)

        print(f'values: {values.shape}; attention: {attention.shape}')

        values = values.reshape(N, V_len, self.num_heads * self.heads_dim)

        out = self.O_proj(values)

        print(f'out: {out.shape}')

        return out



if __name__ == '__main__':
    model = MultiHeadAttention(embed_dim=512, num_heads=8)
    Q = T.randn(32, 128, 512)
    K = T.randn(32, 100, 512)
    V = T.randn(32, 164, 512)
    out = model.forward(Q, K, V)
