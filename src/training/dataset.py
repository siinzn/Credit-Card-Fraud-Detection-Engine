import pandas as pd
from utils import load_dataset, root_dir

def join_dataset(file1, file2):
    #combine dataset
    combined_dataset = pd.concat([file1, file2], ignore_index=True)

    #parse timestam and sort it
    combined_dataset['trans_date_trans_time'] = pd.to_datetime(combined_dataset['trans_date_trans_time'])
    combined_dataset = combined_dataset.sort_values('trans_date_trans_time').reset_index(drop=True)

    #drop the first column which is just index, NOTE - its called Unnamed: 0 if the column doesnt have a name
    combined_dataset = combined_dataset.drop(columns=['Unnamed: 0'])
    return combined_dataset

if __name__  == "__main__":
    train_file_path = root_dir / "data" / "raw" / "fraudTrain.csv"
    test_file_path = root_dir / "data" / "raw" / "fraudTest.csv"
    combined_dataset_path = root_dir / "data" / "raw" / "fraud_combined.csv"

    train_df = load_dataset(filepath=train_file_path)
    test_df = load_dataset(filepath=test_file_path)

    print(train_df.shape)
    print(test_df.shape)

    #save data
    combined_dataset = join_dataset(train_df, test_df)
    combined_dataset.to_csv(combined_dataset_path, index=False)
    print(combined_dataset.head())