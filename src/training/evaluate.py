import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, precision_score, recall_score

def evaluate(y_true, y_score, threshold=None):
    PR_AUC = average_precision_score(y_true=y_true, y_score=y_score)
    ROC_AUC = roc_auc_score(y_true, y_score)
        
    print(f"PR_AUC: {PR_AUC}")
    print(f"ROC_AUC: {ROC_AUC}\n")
