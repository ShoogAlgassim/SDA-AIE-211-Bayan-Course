"""Lab 2 starter: scaled dot-product attention and multi-head attention."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, mask=None, return_weights=False):
    """Compute scaled dot-product attention."""

    d_k = q.size(-1)

    # Q × K^T
    scores = torch.matmul(q, k.transpose(-2, -1))

    # Scale by sqrt(d_k)
    scores = scores / math.sqrt(d_k)

    # Apply mask before softmax
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Convert scores to attention probabilities
    weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(weights, v)

    if return_weights:
        return output, weights

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, return_weights=False):
        batch_size = q.size(0)

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)

        k = k.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)

        v = v.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        if return_weights:
            context, weights = attention(
                q, k, v, mask, return_weights=True
            )
        else:
            context = attention(q, k, v, mask)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        output = self.out_proj(context)

        if return_weights:
            return output, weights

        return output