"""
custom_encoder.py  (OPTIONAL — build this if you want to learn transformers from scratch)

WHAT TO WRITE HERE (per your notes' "Create Encoder" path):
  Text -> Tokenizer -> Token IDs -> Token Embedding layer -> + Positional info
       -> [Self-Attention -> Feed-Forward -> Residual + LayerNorm] x (2-3 layers)
       -> Pooling -> Sentence embedding

Suggested components to implement:
- `TokenEmbedding`: nn.Embedding(vocab_size, d_model) for token IDs.
- `PositionalEncoding`: either learned positional embeddings or sinusoidal, added to
  token embeddings (transformers have no inherent notion of order without this).
- `MultiHeadSelfAttention`: implement Q, K, V projections, scaled dot-product attention
  (softmax(QK^T / sqrt(d_k)) V), and multi-head concat + output projection.
  Use config.yaml's `num_attention_heads` (default 4).
- `FeedForward`: two linear layers with a nonlinearity (e.g. GELU) in between.
- `TransformerEncoderLayer`: self-attention -> residual + LayerNorm -> feed-forward ->
  residual + LayerNorm. Stack `num_layers` of these (default 2-3 per your notes).
- `CustomEncoder`: ties the above together, takes token IDs -> returns per-token
  representations, then applies pooling (mean over non-padded tokens, or use a
  learned [CLS]-like token) to produce a single sentence embedding of size
  config.yaml's `embedding_dim` (128).

This mirrors what BERT does internally — building it yourself is how you check you
understand attention, Q/K/V, positional encoding, and pooling conceptually.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dict_size: int, d_model: int, string_max_len: int = 512):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=dict_size, embedding_dim=d_model) #token embedding layer
        self.pos_embed = nn.Embedding(num_embeddings=string_max_len, embedding_dim=d_model) #positional embedding layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_vect = torch.arange(0, x.size(1), dtype=torch.long, device=x.device) #takes num of cols and creates position vector as [0,1,2...]
        pos_tensor = self.pos_embed(pos_vect)
        pos_tensor = pos_tensor.unsqueeze(0)
        embed_tensor = self.embed(x)
        return embed_tensor + pos_tensor


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.q = nn.Linear(in_features=d_model, out_features=d_model)
        self.k = nn.Linear(in_features=d_model, out_features=d_model)
        self.v = nn.Linear(in_features=d_model, out_features=d_model)
        self.o = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        q = self.q(x).view(batch_size, x.size(1), self.num_heads, self.d_k).transpose(1,2)
        k = self.k(x).view(batch_size, x.size(1), self.num_heads, self.d_k).transpose(1,2)
        v = self.v(x).view(batch_size, x.size(1), self.num_heads, self.d_k).transpose(1,2)
        a = q.matmul(k.transpose(2,3)) / math.sqrt(self.d_k)
        mask = attention_mask.unsqueeze(1).unsqueeze(1)
        a = torch.where(mask == 0, torch.float("-inf"), a)
        a = torch.softmax(a, dim=-1)
        a = a.matmul(v).transpose(1,2).contiguous().view(batch_size, x.size(1), self.d_model)
        return self.o(a)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.expansion = nn.Linear(in_features=d_model, out_features=d_ff)
        self.reconstruct = nn.Linear(in_features=d_ff, out_features=d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = self.expansion(x)
        activated = self.activation(w1)
        w2 = self.reconstruct(activated)
        return w2


class PreLNLayerMHAFFN(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        #maybe look into dropout layers later idk

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x_n = self.norm1(x)
        x_mha = self.mha(x_n, attention_mask)
        x_mid = x_mha + x
        x_n2 = self.norm2(x_mid)
        x_ffn = self.ff(x_n2)
        return x_ffn + x_mid

def pooling(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand_as(x)
    sum_x = torch.sum(x * mask, dim=1)
    count = torch.clamp(torch.sum(attention_mask, dim=1), min=1e-9)
    return sum_x / count

class CustomEncoder(nn.Module):
    """From-scratch transformer text encoder producing sentence embeddings."""

    def __init__(
        self,
        dict_size: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 512,
        string_max_len: int = 512,
    ):
        super().__init__()
        self.dict_size = dict_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.string_max_len = string_max_len
        self.pos_enc = PositionalEncoding(dict_size, d_model, string_max_len)
        self.layers = nn.ModuleList([PreLNLayerMHAFFN(d_model, num_heads, d_ff) for x in range(num_layers)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns a [batch_size, d_model] sentence embedding per input."""
        embedding = self.pos_enc(input_ids)
        for layer in self.layers:
            embedding = layer(embedding, attention_mask)
        return pooling(embedding, attention_mask) #collapses the seq_len dimension in (batch_size, seq_len, d_model) to (batch_size, d_model)
