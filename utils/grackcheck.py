"""Finite-difference gradient checking.

Implement:
    numerical_grad(f, x, h=1e-5) -> np.ndarray
        Central differences (f(x + h e_i) - f(x - h e_i)) / (2h) for every entry of x.
        x may have any shape: iterate with np.nditer, perturb in place, then restore.
    check_grad(f, grad_f, x, h=1e-5, rtol=1e-4, atol=1e-6) -> float
        Compare the analytic grad_f(x) with numerical_grad(f, x). Return the maximum
        relative error and raise AssertionError if it exceeds the tolerance.

Notes:
    Use float64 throughout; float32 makes finite differences too noisy.
    Relative error |a - n| / max(|a|, |n|, tiny) is more informative than absolute error.
    Used by the tests for functional, mlp_numpy, logistic_regression and layers.
"""