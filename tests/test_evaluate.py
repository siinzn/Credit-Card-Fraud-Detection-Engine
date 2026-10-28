import numpy as np
from src.training.evaluate import evaluate
from src.training.dataset import temporal_split
from utils import load_dataset, root_dir

# simple checks

y_true= [1, 0, 1, 1, 0, 1, 0, 0, 1]

y_score_t1 = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
y_score_t2 = np.random.rand(len(y_true))
y_score_t3 = np.zeros(len(y_true))

print("Test 1")
evaluate(y_true=y_true, y_score=y_score_t1)
print("Test 2")
evaluate(y_true=y_true, y_score=y_score_t2)
print("Test 3")
evaluate(y_true=y_true, y_score=y_score_t3)

# data test

test_file_path = root_dir / "data" / "raw" / "fraudTest.csv"
combined_dataset_path = root_dir / "data" / "raw" / "fraud_combined.csv"

test_df = load_dataset(filepath=test_file_path)
combined_dataset = load_dataset(filepath=combined_dataset_path)

train, value, test = temporal_split(test_df=test_df, combined_df=combined_dataset)

y_true_values = value['is_fraud'].values
y_score_values = np.random.rand(len(y_true_values))
print("Test 4")
evaluate(y_true=y_true_values, y_score=y_score_values)