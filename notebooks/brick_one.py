from utils import load_dataset, root_dir
import pandas as pd

train_file_path = root_dir / "data" / "raw" / "fraudTrain.csv"
test_file_path = root_dir / "data" / "raw" / "fraudTest.csv"
combined_file_path = root_dir / "data" / "raw" / "fraud_combined.csv"

train_df = load_dataset(filepath=train_file_path)
test_df = load_dataset(filepath=test_file_path)
combined_df = pd.read_csv(combined_file_path, parse_dates=['trans_date_trans_time'])

# check overall fraud rate per dataset
print(f"Train data fraud rate: {train_df['is_fraud'].mean() * 100}\n")
print(f"Test data fraud rate: {test_df['is_fraud'].mean() * 100}\n")
print(f"Combined data fraud rate: {combined_df['is_fraud'].mean() * 100}\n")

# check fraud rate per hour
combined_df['hour'] = combined_df['trans_date_trans_time'].dt.hour
print(f"{combined_df.groupby('hour')['is_fraud'].mean() * 100}\n")

# check fraud rate per category
print(f"{combined_df.groupby('category')['is_fraud'].mean().sort_values(ascending=True) * 100}\n")

# transactions per card
print(f"{combined_df['cc_num'].value_counts()}\n")

"""
IGNORE BELOW, ONLY FOR COMPARISON IN THE END
Train data has 0.57% fraud rate
Test data has 0.38% fraud rate
Combined data has 0.52% fraud rate

*These are fraud rate per hour*

0     1.356854
1     1.348443
2     1.304362
3     1.317084
4     0.101772
5     0.133138
6     0.089395
7     0.119401
8     0.097524
9     0.101277
10    0.086207
11    0.098056
12    0.090038
13    0.100543
14    0.107424
15    0.107022
16    0.102875
17    0.100520
18    0.118020
19    0.112380
20    0.105285
21    0.107747
22    2.601447
23    2.546349

These are fraud rate per category

health_fitness    0.150955
home              0.151032
food_dining       0.156813
kids_pets         0.187971
entertainment     0.217719
personal_care     0.222931
travel            0.269170
grocery_net       0.269737
misc_pos          0.281890
gas_transport     0.410575
shopping_pos      0.634375
grocery_pos       1.264537
misc_net          1.303859
shopping_net      1.592713

transactions per card

30270432095985         4392
6538441737335434       4392
4642255475285942       4386
4364010865167176       4386
6538891242532018       4386

"""