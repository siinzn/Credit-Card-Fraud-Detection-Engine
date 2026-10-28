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

def temporal_split(test_df, combined_df):
    # first i get the test start, if i have that then i can split combined into two 1st part and test part. now split the 1st part into two train and val. val is just 15% of 1st part

    test_df['trans_date_trans_time'] = pd.to_datetime(test_df['trans_date_trans_time']) # parse datetime for test
    combined_df['trans_date_trans_time'] = pd.to_datetime(combined_df['trans_date_trans_time']) # parse for combined_df csv
    test_df = test_df.sort_values('trans_date_trans_time').reset_index(drop=True) # sort it
    combined_df = combined_df.sort_values('trans_date_trans_time').reset_index(drop=True) # sort it
    test_start = test_df['trans_date_trans_time'].iloc[0]
    pre_test = combined_df[combined_df['trans_date_trans_time'] < test_start]
    min_time = pre_test['trans_date_trans_time'].min()
    val_start = min_time + 0.85 * (test_start - min_time) # get the val start

    train = combined_df[combined_df['trans_date_trans_time'] < val_start]
    value = combined_df[(combined_df['trans_date_trans_time'] >= val_start) & (combined_df['trans_date_trans_time'] < test_start)]
    test = combined_df[combined_df['trans_date_trans_time'] >= test_start]
    """
    #Sanity checks
    print(f"Min of train: {train['trans_date_trans_time'].min()}, Max of train: {train['trans_date_trans_time'].max()}")
    print(f"Min of value: {value['trans_date_trans_time'].min()}, Max of value: {value['trans_date_trans_time'].max()}")
    print(f"Min of test: {test['trans_date_trans_time'].min()}, Max of test: {test['trans_date_trans_time'].max()}")

    # Fraud count checks
    print(f"Combined Fraud: {(combined_df['is_fraud'] == 1).sum()}")
    print(f"Train fraud count: {(train['is_fraud'] == 1).sum()}")
    print(f"Value fraud count: {(value['is_fraud'] == 1).sum()}")
    print(f"Test fraud count: {(test['is_fraud'] == 1).sum()}")
    """

    return(train,value,test)

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
    temporal_split(test_df=test_df, combined_df=combined_dataset)