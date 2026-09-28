"""
Optimisers as small classes with a step() method.

Implement (each holds a list of parameter arrays and updates them in place from gradients):
    SGD(lr)
    Momentum(lr, beta=0.9)          v = beta v + g;  p -= lr v
    Nesterov(lr, beta=0.9)          v = beta v + g;  p -= lr (g + beta v)
    Adam(lr, beta1=0.9, beta2=0.999, eps=1e-8)
        m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g^2;  bias-correct by 1 - b^t
    AdamW(lr, beta1, beta2, eps, weight_decay)
        decoupled decay: p -= lr * weight_decay * p, separately from the Adam step
    warmup_cosine(step, warmup, total, lr_max, lr_min) -> learning rate

Notes:
    Write a NumPy version (for mlp_numpy) and optionally a torch version (update p.data
    under torch.no_grad()) and compare trajectories against torch.optim.
"""