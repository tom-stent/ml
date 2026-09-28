"""
Random forests: bagging plus random feature subsampling, built on DecisionTree.

Implement:
    class RandomForest(task="classification", n_trees=100, max_features="sqrt",
                       max_depth=None, oob_score=False, seed=None)
        fit: for each tree, draw a bootstrap sample (with replacement) and fit a
             DecisionTree with max_features set; store the in-bag indices
        predict: average the trees' probabilities (classification) or predictions (regression)
        oob_score_: predict each point using only the trees that didn't see it
        feature_importances_ (optional): permutation importance on out-of-bag data

Notes:
    About 36.8% of points are out of bag for each tree, since (1 - 1/n)^n -> 1/e.
    max_features=None gives plain bagging; compare the two in a notebook.
"""