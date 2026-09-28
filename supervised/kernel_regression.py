"""
Kernel smoothing and kernel ridge regression.

Implement:
    rbf_kernel(X, Y, gamma), polynomial_kernel(X, Y, degree, coef0), linear_kernel(X, Y)
    class NadarayaWatson(bandwidth)
        predict(x) = sum_i K_h(x - x_i) y_i / sum_i K_h(x - x_i), Gaussian kernel
    class LocalLinear(bandwidth)       # optional: fixes the boundary bias
    class KernelRidge(alpha=1.0, kernel="rbf", gamma=1.0, degree=3, coef0=1.0)
        fit: dual coefficients = solve(K + alpha I, y), via Cholesky (K + alpha I is SPD)
        predict: K(X_new, X_train) @ dual coefficients

Notes:
    Build the kernels from utils.distances.sq_euclidean.
    Kernel ridge with a linear kernel must equal ridge without an intercept: a good test.
"""