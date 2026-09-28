"""
Independent component analysis via FastICA.

Implement:
    whiten(X) -> (Z, whitening_matrix, mean): centre, then PCA whitening so cov(Z) = I
    class FastICA(n_components, max_iter=200, tol=1e-6, seed=None)
        deflation version: for each component, iterate
            w <- mean(z g(w^T z)) - mean(g'(w^T z)) w,  with g = tanh, g' = 1 - tanh^2
            orthogonalise against earlier components (Gram-Schmidt), then normalise
            stop when |<w_new, w_old>| is close to 1
        optional symmetric version: update all rows at once, then W <- (W W^T)^{-1/2} W
        fit(X), transform(X) -> estimated sources; attributes mixing_ and unmixing_

Notes:
    Sources are recovered only up to permutation, sign and scale.
    Keep data as (n_samples, n_features) consistently.
"""