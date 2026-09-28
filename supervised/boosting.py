"""
Boosting: AdaBoost and gradient boosting.

Implement:
    class AdaBoost(n_estimators=50)
        labels in {-1, +1}; weak learner = DecisionTree(max_depth=1) fitted with sample weights
        each round: weighted error eps_t, alpha_t = 0.5 * log((1 - eps_t) / eps_t),
                    w_i *= exp(-alpha_t y_i h_t(x_i)), then renormalise
        predict: sign(sum_t alpha_t h_t(x)); staged_predict for error curves
    class GradientBoosting(loss="squared", n_estimators=100, learning_rate=0.1,
                           max_depth=3, subsample=1.0, seed=None)
        loss in {"squared", "logistic"}
        initialise F_0 with the best constant (mean of y, or the log-odds)
        each round: pseudo-residuals r = -dL/dF (y - F for squared loss,
                    y - sigmoid(F) for logistic loss), fit a regression tree to r,
                    update F += learning_rate * tree(x)
        record train_loss_ every round; staged_predict

Notes:
    For logistic loss, the simple version uses the tree's mean residual in each leaf;
    the Newton leaf value sum(r) / sum(p (1 - p)) is closer to what libraries do.
"""