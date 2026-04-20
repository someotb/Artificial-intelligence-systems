import numpy as np
import pandas as pd
import statsmodels.api as sm
from textblob import TextBlob


def get_tone(text: str):
    if not isinstance(text, str):
        return 0
    return TextBlob(text).sentiment.polarity


def check_for_unic(data: pd.DataFrame) -> None:
    cols_to_check = ["effectiveness", "sideEffects", "condition", "urlDrugName"]
    for col in cols_to_check:
        print(f"--- Столбец: {col} ---")
        print(f"Уникальных значений: {data[col].nunique()}")
        print(f"Примеры значений: {data[col].unique()[:3]}")
        print("\n")


def backward_elimination(X, y, sl=0.05):
    num_cols = X.shape[1]
    remaining_cols = list(range(num_cols))

    X_opt = np.append(arr=np.ones((len(X), 1)).astype(float), values=X, axis=1)

    while True:
        model = sm.OLS(y, X_opt).fit()
        max_p = max(model.pvalues)
        if max_p > sl:
            max_idx = list(model.pvalues).index(max_p)
            X_opt = np.delete(X_opt, max_idx, axis=1)
            remaining_cols.pop(max_idx - 1)  # -1 из-за фиктивного столбца
            print(f"Удалён признак с p-value: {max_p:.4f}")
        else:
            break

    return X_opt, remaining_cols
