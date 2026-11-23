import numpy as np

def add_features(df):
    df = amt_log(df);
    return df

def amt_log(df):
    df['amt_log'] = np.log1p(df['amt'])
    return df
     