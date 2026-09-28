"""
Multidimensional scaling: classical, and (optionally) non-metric.

Implement:
    double_centre(D2) -> B = -0.5 * J D2 J, with J = I - (1/n) 1 1^T and D2 the squared distances
    classical_mds(D, k) -> (X, eigenvalues)
        eigendecompose B with np.linalg.eigh; X = V_k sqrt(Lambda_k) from the top k
        eigenpairs; report negative eigenvalues (a sign of non-Euclidean dissimilarities)
    stress(D, X) -> Kruskal stress-1
Notes:
    The output is only defined up to rotation, reflection and translation, so compare
    results using procrustes.
"""