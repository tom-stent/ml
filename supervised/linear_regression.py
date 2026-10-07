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
    def __init__(
            self, 
            fit_intercept: bool = True, 
            solver: str = "normal",
            num_epochs: int = 1000, 
            tol: float = 1e-6,
            learning_rate: float = 0.01
            ):
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.num_epochs = num_epochs
        self.tol = tol
        self.learning_rate = learning_rate

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

        X, y = np.asarray(X), np.asarray(y)
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights)

        # If solving for intercept, centre first
        if self.fit_intercept:
            if sample_weights is not None:
                X_mean = np.average(X, 0, sample_weights)
                y_mean = np.average(y, 0, sample_weights)
            else:
                X_mean = np.mean(X, 0)
                y_mean = np.mean(y, 0)

            X = X - X_mean
            y = y - y_mean

        self.coef_ = solvers[self.solver](X, y, sample_weights)

        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean.T @ self.coef_
        else:
            self.intercept_ = 0.0

        return self

    def _solve_normal(self, X, y, sample_weights):
        Xw = X if sample_weights is None else X * sample_weights[:, None]
        return np.linalg.solve(
            Xw.T @ X, Xw.T @ y
        )

    def _solve_qr(self, X, y, sample_weights):
        raise NotImplementedError

    def _solve_svd(self, X, y, sample_weights):
        raise NotImplementedError

    def _solve_gd(self, X, y, sample_weights, num_epochs, tol):
        raise NotImplementedError
        
    def predict(self, X):
        if not hasattr(self, "coef_") or not hasattr(self, "intercept_"):
            raise AttributeError("Model must be fit first, use .fit()")

        return X @ self.coef_ + self.intercept_


class RidgeRegression():
    """
    Solver types: normal, qr, svd, gd.
    """
    def __init__(
            self, 
            lam: float = 1.0,
            fit_intercept: bool = True, 
            solver: str = "normal",
            num_epochs: int = 1000, 
            tol: float = 1e-6,
            learning_rate: float = 0.01
            ):
        self.lam = lam
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.num_epochs = num_epochs
        self.tol = tol
        self.learning_rate = learning_rate

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

        X, y = np.asarray(X), np.asarray(y)
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights)

        # If solving for intercept, centre first
        if self.fit_intercept:
            if sample_weights is not None:
                X_mean = np.average(X, 0, sample_weights)
                y_mean = np.average(y, 0, sample_weights)
            else:
                X_mean = np.mean(X, 0)
                y_mean = np.mean(y, 0)

            X = X - X_mean
            y = y - y_mean

        self.coef_ = solvers[self.solver](X, y, sample_weights)

        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean.T @ self.coef_
        else:
            self.intercept_ = 0.0

        return self

    def _solve_normal(self, X, y, sample_weights):
        Xw = X if sample_weights is None else X * sample_weights[:, None]
        return np.linalg.solve(
            Xw.T @ X + self.lam * np.eye(X.shape[1]), Xw.T @ y
        )

    def _solve_qr(self, X, y, sample_weights):
        raise NotImplementedError

    def _solve_svd(self, X, y, sample_weights):
        raise NotImplementedError

    def _solve_gd(self, X, y, sample_weights, num_epochs, tol):
        raise NotImplementedError
        
    def predict(self, X):
        if not hasattr(self, "coef_") or not hasattr(self, "intercept_"):
            raise AttributeError("Model must be fit first, use .fit()")

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