"""
Vectorised pairwise distances and dissimilarities.

Implement (X: (n, d), Y: (m, d) -> (n, m) matrix, with no loops):
    sq_euclidean(X, Y)       # ||x||^2 + ||y||^2 - 2 x.y (clip small negatives from rounding to 0)
    euclidean(X, Y)
    manhattan(X, Y)
    minkowski(X, Y, p)
    mahalanobis(X, Y, S)     # whiten with the Cholesky factor of S, then Euclidean
    cosine_distance(X, Y)    # 1 - cosine similarity
    is_metric(D, tol) -> dict
        checks non-negativity, zero diagonal, symmetry and the triangle inequality
"""