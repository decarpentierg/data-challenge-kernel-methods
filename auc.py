import numpy as np

def auc_score(y_true, y_score):
    """Compute the area under the ROC curve (AUC) given true labels and predicted scores.

    Args:
        y_true (ndarray): Array of true binary labels (-1 or 1).
        y_score (ndarray): Array of predicted logit scores.

    Returns:
        float: The AUC score.
    """
    # sort the predicted scores in descending order and get the corresponding true labels
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    # count the number of true positives and false positives at each threshold
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == -1)
    # calculate the true positive rate (TPR) and false positive rate (FPR) at each threshold
    tpr = tp / tp[-1]
    fpr = fp / fp[-1]
    # calculate the AUC using the trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return auc
