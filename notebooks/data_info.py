from utils import root_dir, load_dataset

train_file_path = root_dir / "data" / "raw" / "fraudTrain.csv"
test_file_path = root_dir / "data" / "raw" / "fraudTest.csv"

train_df = load_dataset(filepath=train_file_path)
test_df = load_dataset(filepath=test_file_path)

print(f"{train_df.shape}, {train_df.columns}")
print(f"{test_df.shape}, {test_df.columns}")

train_total = 0
for value in train_df["is_fraud"]:
    train_total += value

test_total = 0
for value in test_df["is_fraud"]:
    test_total += value

print(f"Rate: {(train_total / train_df.shape[0]) * 100}")
print(f"Rate: {(test_total / test_df.shape[0]) * 100}")