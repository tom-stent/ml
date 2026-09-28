"""
Naive Bayes classifiers, computed in log space.

Implement:
    class GaussianNB(var_smoothing=1e-9)
        fit: class priors, per-class feature means and variances
             (add var_smoothing * the largest feature variance)
        predict_log_proba: log prior + sum of Gaussian log densities, normalised with logsumexp
        predict_proba, predict
    class MultinomialNB(alpha=1.0)
        fit on count features; smoothed log probabilities
            log((count + alpha) / (class_total + alpha * n_features))
        predict: argmax of X @ feature_log_prob.T + class_log_prior

Notes:
    Never multiply raw probabilities, instead sum logs and normalise with logsumexp.
"""