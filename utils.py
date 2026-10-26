import pandas as pd
from pathlib import Path

script_dir = Path(__file__).parent
root_dir = script_dir

def load_dataset(filepath : str):
    return pd.read_csv(filepath)