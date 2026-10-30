import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, precision_score, recall_score

def evaluate(y_true, y_score, threshold=None):
    PR_AUC = average_precision_score(y_true=y_true, y_score=y_score)
    ROC_AUC = roc_auc_score(y_true, y_score)
    return {"pr_auc": PR_AUC, "roc_auc": ROC_AUC}
