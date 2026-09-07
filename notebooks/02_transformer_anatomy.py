"""Lab 2 starter notebook-as-script.
Complete the marked sections, verify numerical equivalence, inspect parameter
accounting, causal masking, attention heads and pad-attention leakage.
"""

import math
import torch
import torch.nn.functional as F

from bayan.attention import attention, MultiHeadAttention


def main():
    torch.manual_seed(42)

    print("=== 1. Numerical equivalence ===")

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    custom = attention(q, k, v)
    reference = F.scaled_dot_product_attention(q, k, v)

    print("Custom matches PyTorch:",
          torch.allclose(custom, reference, atol=1e-6))

    assert torch.allclose(custom, reference, atol=1e-6)

    print("\n=== 2. Attention weights ===")

    _, weights = attention(
        q,
        k,
        v,
        return_weights=True
    )

    print(weights[0, 0])

    print("\n=== 3. Multi-Head Attention ===")

    mha = MultiHeadAttention(
        d_model=16,
        num_heads=2
    )

    x = torch.randn(1, 4, 16)

    mha_output, mha_weights = mha(
        x,
        x,
        x,
        return_weights=True
    )

    print("MHA output shape:", mha_output.shape)
    print("MHA weights shape:", mha_weights.shape)

    print("\n=== 4. Causal mask ===")

    seq_len = 4

    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len)
    )

    _, causal_weights = attention(
        q,
        k,
        v,
        mask=causal_mask,
        return_weights=True
    )

    print("Causal attention matrix:")
    print(causal_weights[0, 0])

    future_mass = torch.triu(
        causal_weights[0, 0],
        diagonal=1
    ).sum()

    print("Attention paid to future tokens:",
          future_mass.item())

    assert torch.allclose(
        future_mass,
        torch.tensor(0.0),
        atol=1e-6
    )

    print("\n=== 5. Pad-attention leakage ===")

    pad_mask = torch.tensor(
        [[1, 1, 1, 0]],
        dtype=torch.float32
    )

    pad_mask = pad_mask[:, None, None, :]

    _, weights_without_mask = attention(
        q,
        k,
        v,
        return_weights=True
    )

    _, weights_with_mask = attention(
        q,
        k,
        v,
        mask=pad_mask,
        return_weights=True
    )

    pad_mass_without = weights_without_mask[..., -1].sum()
    pad_mass_with = weights_with_mask[..., -1].sum()

    print(
        "Pad mass without mask:",
        pad_mass_without.item()
    )

    print(
        "Pad mass with mask:",
        pad_mass_with.item()
    )

    assert torch.allclose(
        pad_mass_with,
        torch.tensor(0.0),
        atol=1e-6
    )


if __name__ == "__main__":
    main()