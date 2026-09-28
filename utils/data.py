"""
Synthetic datasets with known ground truth, for tests and notebooks.

Implement (each takes a seed and returns NumPy arrays):
    make_linear(n, d, noise, seed) -> X, y, true_w, true_b
    make_blobs(n, centers, std, seed) -> X, labels
    make_xor(n, noise, seed) -> X, y
    make_spirals(n, n_classes, noise, seed) -> X, y
    make_gmm(n, means, covs, weights, seed) -> X, labels
    make_sine(n, noise, seed) -> x, y                 # 1D regression, for smoothers
    make_piecewise_signal(n, noise, seed) -> t, y     # signal with jumps, for wavelets
    make_sources(n, seed) -> S                        # sine, square and sawtooth rows, for ICA
    train_test_split(X, y, test_frac, seed) -> X_tr, X_te, y_tr, y_te

Notes:
    Use rng = np.random.default_rng(seed)
"""