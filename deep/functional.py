"""Core activations and losses in NumPy, with manual gradients.

Implement (forward and gradient for each; float64-safe):
    relu, relu_grad
    sigmoid (numerically stable), sigmoid_grad
    tanh_grad
    gelu (tanh approximation), gelu_grad
    logsumexp(x, axis): m + log(sum(exp(x - m))), with m the max along axis
    softmax(x, axis): subtract the max first
    log_softmax(x, axis) = x - logsumexp(x, axis, keepdims=True)
    cross_entropy(logits, y) -> mean loss, y as integer labels
    cross_entropy_grad(logits, y) -> (softmax(logits) - one_hot(y)) / n
    mse, mse_grad

Notes:
    Never compute log(softmax(x)); use log_softmax. Test with logits like [1000, 1001].
"""