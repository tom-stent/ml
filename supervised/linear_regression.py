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