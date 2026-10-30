from utils import root_dir, load_dataset
from src.training.dataset import temporal_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.training.evaluate import evaluate

# logistic regression model

RAW_FEATURES = [
    'amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long'
]

def save_results(results, val):
    result_file = root_dir / "results.md"
    with result_file.open("a", encoding="utf-8") as file:
        legit_accuracy = 1 - val['is_fraud'].mean()
        file.write(
            f"Model | Logistic Regression\n"
            f"PR_AUC | {results['pr_auc']:.6f}\n"
            f"ROC-AUC | {results['roc_auc']:.6f}\n"
            f"Legit Accuracy | {legit_accuracy:.6f}\n"
        )
        
def train_baseline(train, val):
    X_train = train[RAW_FEATURES]
    Y_train = train['is_fraud']

    X_val = val[RAW_FEATURES]
    Y_val = val['is_fraud']
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train_scaled, Y_train)
    score = clf.predict_proba(X_val_scaled)[:, 1]
    return evaluate(Y_val, score)

if __name__ == "__main__":
    test_file_path = root_dir / "data" / "raw" / "fraudTest.csv"
    combined_dataset_path = root_dir / "data" / "raw" / "fraud_combined.csv"

    test_df = load_dataset(filepath=test_file_path)
    combined_dataset = load_dataset(filepath=combined_dataset_path)

    train, val, test = temporal_split(test_df=test_df, combined_df=combined_dataset)
    result = train_baseline(train=train, val=val)
    save_results(results=result, val=val)