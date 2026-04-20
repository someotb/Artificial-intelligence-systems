import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_test = pd.read_csv("../data/lab6_X_test.csv")
X_train = pd.read_csv("../data/lab6_X_train.csv")
y_test = pd.read_csv("../data/lab6_Y_test.csv")
y_train = pd.read_csv("../data/lab6_Y_train.csv")

# Одномерная регрессия
X_train_1d = X_train[["effectiveness"]]
X_test_1d = X_test[["effectiveness"]]

regressor_1d = LinearRegression()
regressor_1d.fit(X_train_1d, y_train)
y_pred_1d = regressor_1d.predict(X_test_1d)

print("Одномерная регрессия")
print(f"R2:   {r2_score(y_test, y_pred_1d):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_1d):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_1d)):.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(X_train_1d, y_train, color="red", label="Real", alpha=0.5)
plt.plot(
    X_train_1d, regressor_1d.predict(X_train_1d), color="blue", label="Regression line"
)
plt.title("1D: Rating vs Target")
plt.xlabel("Rating")
plt.ylabel("Target")
plt.legend()

# Многомерная регрессия
regressor_md = LinearRegression()
regressor_md.fit(X_train, y_train)
y_pred_md = regressor_md.predict(X_test)

print("\nМногомерная регрессия")
print(f"R2:   {r2_score(y_test, y_pred_md):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_md):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_md)):.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test.values, color="red", label="Real", alpha=0.5)
plt.scatter(range(len(y_test)), y_pred_md, color="blue", label="Predicted", alpha=0.5)
plt.title("Multi-D: Real vs Predicted (Test)")
plt.xlabel("Sample index")
plt.ylabel("Target")
plt.legend()
plt.show()
