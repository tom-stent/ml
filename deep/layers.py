"""
Neural-network layers in PyTorch, written without the built-in torch.nn layers.

Implement (subclass nn.Module; use only nn.Parameter and tensor operations):
    Linear(in_features, out_features, bias=True)   # init U(-1/sqrt(in), 1/sqrt(in)), like torch
    LayerNorm(dim, eps=1e-5)                        # over the last dim; learnable gamma, beta
    Dropout(p)                                      # inverted dropout; identity when not self.training
    Embedding(num_embeddings, dim)                  # weight[indices]
    ReLU(), GELU()
    MLPBlock(dim, hidden_mult=4, dropout=0.0)       # Linear -> GELU -> Linear -> Dropout

Notes:
    Autograd supplies the gradients here, the point is correct shapes, initialisation
    and train/eval behaviour. Use the biased variance (unbiased=False) in LayerNorm to
    match torch.
"""