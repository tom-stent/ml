"""
Gaussian mixture model fitted with EM.

Implement:
    log_gaussian_pdf(X, mean, cov) -> (n,), using a Cholesky factorisation (no explicit inverse)
    class GMM(k, max_iter=200, tol=1e-6, reg_covar=1e-6, init="kmeans", seed=None)
        E-step: log responsibilities = log pi_k + log N(x | mu_k, Sigma_k), normalised
                with logsumexp
        M-step: N_k, means, covariances (+ reg_covar * I), weights
        record log_likelihood_history_; stop when the improvement is below tol
        predict(X), predict_proba(X), sample(n)
        bic(X) = -2 log L + p log n, with p the number of free parameters

Notes:
    Work in log space throughout. The reg_covar ridge stops components collapsing
    onto single points.
"""