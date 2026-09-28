"""Evaluation metrics, written from scratch.

Implement:
    mse(y, y_hat), mae(y, y_hat), r2(y, y_hat)
    accuracy(y, y_pred)
    log_loss(y, p, eps=1e-12)              # binary and multiclass
    confusion_matrix(y, y_pred, n_classes)
    precision_recall_f1(y, y_pred)         # binary
    roc_auc(y, scores)                     # via ranks (Mann-Whitney formulation)
    kfold_indices(n, k, seed) -> list of (train_idx, val_idx)

Notes:
    Clip probabilities in log_loss to avoid log(0).
    AUC = P(score of a random positive > score of a random negative),
    handle ties with average ranks.
"""