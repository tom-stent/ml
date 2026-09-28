"""Principal component analysis via the SVD.

Implement:
    class PCA(n_components=None, whiten=False)
        fit(X): centre (store mean_), then SVD of the centred data X_c = U S V^T
        components_ = V^T[:k], explained_variance_ = S^2 / (n - 1),
        explained_variance_ratio_
        transform(X), inverse_transform(Z), fit_transform(X)
        whiten=True: divide the scores by sqrt(explained_variance_)

Notes:
    Use np.linalg.svd(X_c, full_matrices=False); don't form X^T X, which squares the
    condition number. Fix a sign convention (e.g. make the largest-magnitude entry of each
    component positive) so results are comparable with sklearn.
"""