# Experiment Results

Model - Logistic Regression | PR_AUC - 0.188090 | ROC-AUC - 0.833460| Legit Accuracy - 0.994151 
Model - XGBoost | PR_AUC - 0.184470 | ROC-AUC - 0.800713| Legit Accuracy - 0.994151 | Best iteration - 42 #no scale_pos_weight
Model - XGBoost with scale_pos_weight | PR_AUC - 0.270908 | ROC-AUC - 0.960185| Legit Accuracy - 0.994151 | Best iteration - 1 #full weight
Model - XGBoost with scale_pos_weight | PR_AUC - 0.292143 | ROC-AUC - 0.963848| Legit Accuracy - 0.994151 | Best iteration - 1 #sqrt of neg_pos_ratio
Model - XGBoost with scale_pos_weight | PR_AUC - 0.259790 | ROC-AUC - 0.957345| Legit Accuracy - 0.994151 | Best iteration - 0 #no scale_pos_weight but with new setup

# XGBoost Model with sqrt of scale_pos_weight wins since it has the highest PR_AUC. Will be used further. 
