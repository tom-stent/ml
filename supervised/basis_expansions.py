"""
Basis expansions for regression: polynomials and cubic splines.

Implement:
    polynomial_features(x, degree) -> (n, degree + 1) design matrix, for 1D input
    cubic_spline_basis(x, knots) -> columns [1, x, x^2, x^3, (x - k_1)_+^3, ..., (x - k_K)_+^3]
    natural_spline_basis(x, knots) -> K columns, linear beyond the boundary knots
        (ESL eqs 5.4-5.5: N_1 = 1, N_2 = x, N_{k+2} = d_k(x) - d_{K-1}(x))
    class BasisRegression(basis="cubic", degree=3, knots=None, alpha=0.0)
        basis in {"poly", "cubic", "natural"}
        fit(x, y): build the basis, then least squares (ridge if alpha > 0)
        predict(x)

Notes:
    The truncated power basis is badly conditioned; fine for learning, but libraries use
    B-splines so implement that too. Place knots at quantiles of x by default.
"""