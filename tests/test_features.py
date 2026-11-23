import pandas as pd
from src.training.features import add_features
import pytest


def test_add_features_returns_same_length():
    data = {
        "amt": [10.50, 125.99, 3.75],
        "trans_date_trans_time": [
            "2020-01-01 12:30:00",
            "2020-01-02 18:45:30",
            "2020-01-03 09:15:10",
        ],
        "lat": [40.7128, 51.5074, 48.8566],
        "long": [-74.0060, -0.1278, 2.3522],
        "merch_lat": [40.7138, 51.5174, 48.8666],
        "merch_long": [-74.0160, -0.1378, 2.3622],
        "dob": ["1990-05-12", "1985-11-23", "2001-07-08"],
        "category": ["grocery_pos", "shopping_net", "food_dining"],
        "is_fraud": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    result = add_features(df)
    assert len(result) == len(df), "add_features changed the row count"

def test_amt_log():
    data = {
        "amt": [0, 1.718281828]
    }
    df = pd.DataFrame(data)
    result = add_features(df)
    assert result['amt_log'].iloc[0] == 0 and result['amt_log'].iloc[1] == pytest.approx(1.0)

def test_hour_cyclic():
    data = {
    "trans_date_trans_time": [
            "2020-01-01 00:00:00",
            "2020-01-01 06:00:00",
            "2020-01-01 12:00:00",
        ],
    }
    df = pd.DataFrame(data=data)
    result = add_features(df)
    assert result['hour_sin'].iloc[0] == pytest.approx(0) and result['hour_cos'].iloc[0] == pytest.approx(1)
    assert result['hour_sin'].iloc[1] == pytest.approx(1) and result['hour_cos'].iloc[1] == pytest.approx(0)
    assert result['hour_cos'].iloc[2] == pytest.approx(-1)

def test_distance():
    data = {
        "lat": [40.7128, 51.5074, 48.8566],
        "long": [-74.0060, -0.1278, 2.3522],
        "merch_lat": [40.7138, 51.5174, 48.8666],
        "merch_long": [-74.0160, -0.1378, 2.3622],
    }
    df=pd.DataFrame(data)
    result=add_features(df)
    assert result["dist_km"].iloc[0] == pytest.approx(0.850143)
    assert result["dist_km"].iloc[1] == pytest.approx(1.309703)
    assert result["dist_km"].iloc[2] == pytest.approx(1.331004)
