import pandas as pd
import numpy as np
import math

def add_features(df):
    df = amt_log(df)
    df = hour_cyclic(df)
    df = distance(df)
    df = age(df)
    return df

def amt_log(df):
    df['amt_log'] = np.log1p(df['amt'])
    return df

def hour_cyclic(df):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    hour_values  = df['trans_date_trans_time'].dt.hour
    df['hour_sin'] = np.sin((2 * math.pi * hour_values )/24)
    df['hour_cos'] = np.cos((2 * math.pi * hour_values )/24)
    return df

def distance(df):
    r_lat = np.radians(df['lat'])
    r_long = np.radians(df['long'])
    r_merc_lat = np.radians(df['merch_lat'])
    r_merc_long = np.radians(df['merch_long'])

    dlat = r_merc_lat - r_lat
    dlong = r_merc_long - r_long
    a = np.square(np.sin(dlat/2)) + np.cos(r_lat) * np.cos(r_merc_lat) * np.square(np.sin(dlong/2))
    df['dist_km'] = 2 * 6371 * np.arcsin(np.sqrt(a))
    return df 

def age(df):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days / 365.25
    return df

def build_category_map(train_df):
    categories = sorted(train_df['category'].dropna().unique())
    return {
        category: idx for idx, category in enumerate(categories)
    }

def apply_category_map(df, mapping):
    df['category_code'] = df['category'].map(mapping)
    df['category_code'] = df['category_code'].fillna(-1.0)
    return df

def add_velocity_features(df):
    assert df['unix_time'].is_monotonic_increasing and df.index.equals(pd.RangeIndex(len(df))), "DataFrame must be sorted by unix_time ascending with a clean 0..N-1 index"
    #seconds since last feature
    secs_since_last(df)
    #amount by mean of previous amounts
    amt_ratio(df)
    #transactions in the past 1h
    txn_count(df,1)
    #transactions in the past 24h
    txn_count(df,24)
    return df

def secs_since_last(df):
    df['secs_since_last'] = df.groupby("cc_num")['unix_time'].diff()
    df['secs_since_last'] = df['secs_since_last'].fillna(-1)
    return df

def amt_ratio(df):
    amt_sum = df.groupby("cc_num")['amt'].cumsum()
    amt_count = df.groupby("cc_num")['amt'].cumcount()
    earlier_total = amt_sum - df['amt']
    earlier_mean = earlier_total / amt_count
    df['amt_ratio'] = np.where((earlier_total == 0) | (amt_count == 0), 1.0, df['amt'] / earlier_mean)
    return df

def txn_count(df, hours):
    #rolling returns the counts in card-by-card order with timestamp labels, so you rebuild the real row numbers in that same order and pair them with the counts.
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    # rolling creates a sliding window per card, closed=left excludes current row, window is [t - hours, t)
    rolling_counts = (
        df.groupby("cc_num")
          .rolling(f"{hours}h", on="trans_date_trans_time", closed="left")["amt"]
          .count()
    )
    row_order = df.sort_values("cc_num", kind="stable").index
    counts = pd.Series(rolling_counts.values, index=row_order)
    df[f"txn_count_{hours}h"] = counts.fillna(0)
    return df

