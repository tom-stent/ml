"""
Linear regression: OLS, ridge and Lasso with several solvers.

Implement:
    class LinearRegression(fit_intercept=True, solver="normal", lr=0.01, n_iter=1000)
        solver in {"normal", "qr", "gd"}
        fit(X, y), predict(X); attributes coef_, intercept_
        "normal": solve (X^T X) w = X^T y with np.linalg.solve (never np.linalg.inv)
        "qr":     X = QR, then solve R w = Q^T y (swap in numlinalg.qr later)
        "gd":     gradient descent on the MSE, gradient (2/n) X^T (Xw - y);
                  record loss_history_
    class Ridge(alpha=1.0, fit_intercept=True)
        closed form w = (X^T X + alpha I)^{-1} X^T y, computed with np.linalg.solve

    class Lasso...

Notes:
    Handle the intercept by centring X and y (then recovering the intercept), not by
    appending a column of ones, so that ridge doesn't penalise the intercept.
    Optional extra: standard errors from sigma^2 (X^T X)^{-1}, with sigma^2 = RSS / (n - p).
"""

import numpy as np
from utils.data import make_linear


class LinearRegression():
    """
    Solver types: normal, qr, svd, gd.
    """
    def __init__(self, fit_intercept: bool = True, solver: str = "normal"):
        self.fit_intercept = fit_intercept
        self.solver = solver

    def fit(self, X, y, sample_weights=None):

        solvers = {
            "normal": self._solve_normal,
            "qr": self._solve_qr,
            "svd": self._solve_svd,
            "gd": self._solve_gd
        }
        if self.solver not in solvers:
            raise ValueError(
                f"Unknown solver: {self.solver}. Choose from: {list(solvers)}."
            )

        if self.fit_intercept:
            X_mean = np.mean(X, 0)
            y_mean = np.mean(y, 0)

            X = X - X_mean
            y = y - y_mean

        if sample_weights is not None:
            if len(sample_weights) != X.shape[0]:
                raise ValueError("Sample weights incorrect length.")
            X *= sample_weights
            y *= sample_weights

        self.coef_ = solvers[self.solver](X, y)

        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean.T @ self.coef_
        else:
            self.intercept_ = 0.0

        return self

    def _solve_normal(self, X, y):
        return np.linalg.solve(X.T @ X, X.T) @ y

    def _solve_qr():
        pass

    def _solve_svd():
        pass

    def _solve_gd(num_epochs: int, tol: float):
        pass
            


    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            print("Model must be fit first, use .fit()")
            # what here to break out of function?

        return X @ self.coef_ + self.intercept_


class RidgeRegression():
    """
    Solver types: normal, qr, svd, gd.
    """
    def __init__(
            self, 
            fit_intercept: bool = True, 
            lam: float = 1.0,
            solver: str = "normal", 
        ):
        self.fit_intercept = fit_intercept
        self.lam = lam
        self.solver = solver

    def fit(self, X, y, sample_weights=None):

        solvers = {
            "normal": self._solve_normal,
            "qr": self._solve_qr,
            "svd": self._solve_svd,
            "gd": self._solve_gd
        }
        if self.solver not in solvers:
            raise ValueError(
                f"Unknown solver: {self.solver}. Choose from: {list(solvers)}."
            )

        if self.fit_intercept:
            X_mean = np.mean(X, 0)
            y_mean = np.mean(y, 0)

            X = X - X_mean
            y = y - y_mean

        self.coef_ = solvers[self.solver](X, y)

        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean.T @ self.coef_
        else:
            self.intercept_ = 0.0

        return self

    def _solve_normal(self, X, y):
        _, d = X.shape
        return np.linalg.solve(X.T @ X + self.lam * np.eye(d), X.T) @ y

    def _solve_qr():
        pass

    def _solve_svd():
        pass

    def _solve_gd():
        pass

    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            print("Model must be fit first, use .fit()")
            # what here to break out of function?

        return X @ self.coef_ + self.intercept_


class ElasticNetRegression():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

class LassoRegression():
    def __init__(self):
        pass

    def fit():
        pass

    def predict():
        pass