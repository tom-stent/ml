"""
A decoder-only transformer (a small GPT) built from your own layers.

Implement:
    class TransformerBlock(d_model, n_heads, dropout)
        pre-norm: x = x + attn(ln1(x));  x = x + mlp(ln2(x))
    class GPT(vocab_size, block_size, d_model, n_heads, n_layers, dropout)
        token embedding + learned positional embedding -> blocks -> final LayerNorm -> LM head
        (optionally tie the LM head weights to the token embedding)
        forward(idx, targets=None) -> (logits, loss)
        generate(idx, max_new_tokens, temperature=1.0, top_k=None)
    class CharTokenizer: encode and decode for a character-level dataset

Notes:
    Follow Andrej Karpathy's nanoGPT and "Let's build GPT" for structure.
"""