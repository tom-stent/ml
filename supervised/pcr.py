"""
Principal components regression.

Implement:
    class PCR(n_components)
        fit(X, y): standardise X, fit unsupervised.pca.PCA, regress y on the first
            n_components scores with LinearRegression, then map the coefficients back
            to the original features: beta = V_k gamma (and undo the standardisation)
        predict(X)
    select_n_components(X, y, k_folds=5) -> the n_components with the lowest CV MSE

Notes:
    With n_components = n_features, PCR equals OLS: a good test.
    Contrast with ridge: PCR drops low-variance directions entirely, while ridge shrinks
    them by d_j^2 / (d_j^2 + alpha).
"""