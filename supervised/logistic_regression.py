"""
Binary and multiclass logistic regression, trained by gradient descent.

Implement:
    sigmoid(z)       # numerically stable: branch on the sign of z
    class LogisticRegression(lr=0.1, n_iter=1000, l2=0.0, fit_intercept=True)
        fit(X, y) with y in {0, 1}; loss = mean log-loss + (l2 / 2) ||w||^2
        gradient X^T (sigmoid(Xw) - y) / n + l2 * w; record loss_history_
        predict_proba(X), predict(X, threshold=0.5)
    class SoftmaxRegression(lr=0.1, n_iter=1000, l2=0.0)
        fit(X, y) with integer labels; one-hot targets; gradient X^T (P - Y) / n + l2 * W
        predict_proba(X) using a stable softmax (subtract the row max), predict(X)

Notes:
    Implement Newton's method (IRLS) for the binary case, with Hessian X^T S X,
    S = diag(p (1 - p)).
    Unregularised fits on separable data diverge so tests should use l2 > 0 or
    non-separable data.
"""