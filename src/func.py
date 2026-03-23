import pandas as pd
from textblob import TextBlob

def get_tone(text: str):
    if not isinstance(text, str): return 0
    return TextBlob(text).sentiment.polarity

def check_for_unic(data: pd.DataFrame) -> None:
    cols_to_check = ["effectiveness", "sideEffects", "condition", "urlDrugName"]
    for col in cols_to_check:
        print(f"--- Столбец: {col} ---")
        print(f"Уникальных значений: {data[col].nunique()}")
        print(f"Примеры значений: {data[col].unique()[:3]}")
        print("\n")