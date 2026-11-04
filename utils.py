from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)


def compute_metrics(model, X, y_true):
    """Return (metrics dict, predictions). y_true must be label-encoded (0..k-1)."""
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    n_classes = len(model.classes_)
    avg = "binary" if n_classes == 2 else "weighted"

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
        else:
            auc = roc_auc_score(y_true, proba, multi_class="ovr",
                                average="weighted", labels=model.classes_)
    except ValueError:  # e.g. test file is missing a class
        auc = float("nan")

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred
