from utils import root_dir, load_dataset
from src.training.dataset import temporal_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.training.evaluate import evaluate
from xgboost import XGBClassifier 

# logistic regression model

RAW_FEATURES = [
    'amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long'
]

def save_results(results, val, model_name, extra_results=None):
    result_file = root_dir / "results.md"
    with result_file.open("a", encoding="utf-8") as file:
        legit_accuracy = 1 - val["is_fraud"].mean()
        file.write(
            f"Model | {model_name}\n"
            f"PR_AUC | {results['pr_auc']:.6f}\n"
            f"ROC-AUC | {results['roc_auc']:.6f}\n"
            f"Legit Accuracy | {legit_accuracy:.6f}\n"
        )
        if extra_results:
            for name, value in extra_results.items():
                file.write(f"{name} | {value}\n")
        file.write("\n")
        
def train_baseline(train, val):
    #we use train and val data since, val is future transaction, so the model learns from train and then validates its predicts with val data
    X_train = train[RAW_FEATURES]
    Y_train = train['is_fraud']

    X_val = val[RAW_FEATURES]
    Y_val = val['is_fraud']
    scaler = StandardScaler()
    scaler.fit(X_train) # fit is use learn something from the given data. it calculates stats like mean and standard deviation of each feature
    # we dont use fit on the val data since the model cannot know about it. it should learn from train data and then apply that knowledge with val
    X_train_scaled = scaler.transform(X_train) # transform applies what was learn with fit earlier
    X_val_scaled = scaler.transform(X_val)

    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train_scaled, Y_train)
    score = clf.predict_proba(X_val_scaled)[:, 1]
    return evaluate(Y_val, score)

def train_xgboost(train, val):
    X_train = train[RAW_FEATURES]
    Y_train = train['is_fraud']

    X_val = val[RAW_FEATURES]
    Y_val = val['is_fraud']

    """
    binary:logistic outputs probabilites from 0-1, 0.02 legitimate, 0.87 - fraud
    early stopping rounds - xgboost builds trees on at a time, after each tree it checks performance on validation set. if the performance is not improving
    training is stopping so it prevents overfitting and avoid building unwanted trees
    best_iteration - tells which tree numbner had the best val data performance
    """
    xgb = XGBClassifier(objective="binary:logistic", early_stopping_rounds = 20)
    xgb.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=False)
    score = xgb.predict_proba(X_val)[:,1] #[:,1] means - : - select every row, 1 - get the first index which is the fraud probability
    metrics = evaluate(Y_val, score)
    trees = xgb.best_iteration
    return {
    "score": score,
    "best_iteration": trees,
    "metrics": metrics
}

if __name__ == "__main__":
    test_file_path = root_dir / "data" / "raw" / "fraudTest.csv"
    combined_dataset_path = root_dir / "data" / "raw" / "fraud_combined.csv"

    test_df = load_dataset(filepath=test_file_path)
    combined_dataset = load_dataset(filepath=combined_dataset_path)

    train, val, test = temporal_split(test_df=test_df, combined_df=combined_dataset)
    result_baseline = train_baseline(train=train, val=val)
    save_results(results=result_baseline, val=val, model_name="Logistic Regression")
    result_xgboost = train_xgboost(train=train, val=val)
    save_results(
        results=result_xgboost["metrics"],
        val=val,
        model_name="XGBoost",
        extra_results={
            "Best iteration": result_xgboost["best_iteration"]
        }
    )