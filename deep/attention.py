"""
Scaled dot-product attention and multi-head attention in PyTorch.

Implement:
    scaled_dot_product_attention(q, k, v, mask=None) -> (out, weights)
        scores = q @ k.transpose(-2, -1) / sqrt(d_k); set masked scores to -inf, then softmax
    causal_mask(T) -> boolean (T, T) lower-triangular mask
    class MultiHeadAttention(d_model, n_heads, dropout=0.0, causal=False)
        one fused QKV projection (d_model -> 3 * d_model), split into heads:
        (B, T, C) -> (B, n_heads, T, head_dim); attend; merge the heads; output projection

Notes:
    Get the reshapes and transposes right.
    Dividing by sqrt(d_k) keeps the logits' variance near 1, so the softmax doesn't saturate.
"""