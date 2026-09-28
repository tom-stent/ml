"""A multi-layer perceptron with manual backpropagation, in pure NumPy.

Implement:
    class MLP(layer_sizes, activation="relu", seed=None)
        parameters: a list of (W, b); He initialisation W ~ N(0, 2 / fan_in)
        forward(X) -> logits, caching pre-activations and activations
        backward(dlogits) -> gradients for every W and b, using
            delta_l = (delta_{l+1} propagated back through W_{l+1}) * act'(a_l)
        params_and_grads() for use with deep.optimizers
    train(model, X, y, optimizer, epochs, batch_size, seed) -> loss history
        mini-batch loop, reshuffling each epoch; cross-entropy from deep.functional

Notes:
    Verify backward() with utils.gradcheck before training anything.
    Fix one convention (X is (n, d) and h = X W + b, say) and derive the gradient
    shapes.
"""