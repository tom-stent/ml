"""
(Optional) Linear soft-margin SVM trained by subgradient descent.

Implement:
    class LinearSVM(C=1.0, lr=0.01, n_epochs=100, seed=None)
        labels in {-1, +1}; objective 0.5 ||w||^2 + C * mean(max(0, 1 - y (w.x + b)))
        subgradient: w - C * mean over margin violators of y_i x_i (and similarly for b)
        fit, decision_function, predict; record objective_history_
    Stretch goal: a kernel SVM via a simplified SMO on the dual problem.

Notes:
    Support vectors are the points with y (w.x + b) <= 1, up to a tolerance.
"""