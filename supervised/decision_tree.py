"""
CART decision trees for classification and regression.

Implement:
    class DecisionTree(task="classification", criterion="gini", max_depth=None,
                       min_samples_split=2, min_samples_leaf=1, max_features=None, seed=None)
        task in {"classification", "regression"}; criterion in {"gini", "entropy", "mse"}
        fit(X, y, sample_weight=None): greedy recursive binary splitting. For each candidate
            feature, sort the values and evaluate thresholds at midpoints; choose the split
            with the largest (weighted) impurity decrease.
        predict(X): route each row to a leaf; leaf value = majority class or mean
        predict_proba(X): class proportions in the leaf
    A small Node dataclass: feature, threshold, left, right, value

Notes:
    max_features enables the random feature subsampling used by random_forest.
    sample_weight is needed by AdaBoost in boosting.py.
    Efficient split search: after sorting, update left/right class counts (or sums and sums
    of squares) incrementally, rather than recomputing the impurity for every threshold.
    Implemen cost-complexity pruning.
"""