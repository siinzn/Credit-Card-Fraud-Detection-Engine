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
