"""Chapter 6 - the transformer architecture, from one neuron to a full block."""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ---------- 6.1 A simple feed-forward network (FlickSageNet) ----------
class FlickSageNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)

model = FlickSageNet(n_features=8)
n_params = sum(p.numel() for p in model.parameters())
print(f"FlickSageNet parameters: {n_params}")  # 8*64+64 + 64*32+32 + 32*1+1 = 2,689

# ---------- 6.4 Self-attention, step by step ----------
seq_length, d_model = 5, 8
movie_embeddings = torch.randn(seq_length, d_model)
W_query, W_key, W_value = (torch.randn(d_model, d_model) for _ in range(3))
Q, K, V = movie_embeddings @ W_query, movie_embeddings @ W_key, movie_embeddings @ W_value
scores = Q @ K.T / (d_model ** 0.5)  # scale by sqrt(d_model)
attention_weights = F.softmax(scores, dim=-1)
output = attention_weights @ V
print(f"Attention weights shape: {attention_weights.shape}")  # [5, 5]
print(f"Output shape: {output.shape}")  # [5, 8]

# ---------- Multi-head attention ----------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.W_qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        attn = F.softmax((q @ k.transpose(-2, -1)) / (self.d_head ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_out(out)

# ---------- The full transformer block ----------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))   # attention + residual
        x = x + self.ff(self.norm2(x))           # feed-forward + residual
        return x

block = TransformerBlock(d_model=64, n_heads=8, d_ff=256)
x = torch.randn(2, 10, 64)  # batch of 2 sequences, 10 movies each
print(f"Transformer block output: {block(x).shape}")  # [2, 10, 64]
print("Stack 12-96 of these blocks and you have the body of an LLM.")
