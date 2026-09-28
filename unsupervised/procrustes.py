"""
Procrustes analysis: align one configuration to another.

Implement:
    orthogonal_procrustes(A, B) -> Q minimising ||A Q - B||_F over orthogonal Q
        SVD of A^T B = U S V^T, then Q = U V^T
    procrustes(X, Y, scaling=True) -> (Y_aligned, Q, s, t, residual)
        centre both configurations; optional scale s = trace(S) / ||Y_c||_F^2;
        rotate, scale and translate Y onto X
    Optional: restrict to proper rotations (det Q = +1) by flipping the sign of the last
    singular vector when needed.

Notes:
    Used to test MDS and PCA outputs, which are only unique up to rotation and reflection.
"""