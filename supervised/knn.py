"""
k-nearest neighbours for classification and regression.

Implement:
    class KNNClassifier(k=5)
        fit(X, y) stores the data
        predict(X): majority vote among the k nearest (document your tie-break rule)
        predict_proba(X): class frequencies among the neighbours
    class KNNRegressor(k=5, weights="uniform")
        weights in {"uniform", "distance"}
        predict(X): mean, or inverse-distance-weighted mean, of the neighbours' targets

Notes:
    Compute the distance matrix with utils.distances, then use np.argpartition
    (linear time per row) rather than a full argsort.
    Features must be standardised for distances to be meaningful.
"""