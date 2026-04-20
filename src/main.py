import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_test = pd.read_csv("../data/lab6_X_test.csv")
X_train = pd.read_csv("../data/lab6_X_train.csv")
y_test = pd.read_csv("../data/lab6_Y_test.csv")
y_train = pd.read_csv("../data/lab6_Y_train.csv")

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Первые 5 элементов предсказания:\n{y_pred[0:5]}")


print(f"R2 score: {r2_score(y_test, y_pred):.4f}")  # чем ближе к 1 тем лучше
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")  # Mean Absolute Error
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")  # Mean Square Error
print(
    f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}"
)  # Root Mean Square Error

# Train
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_train)), y_train.values, color="red", label="Real", alpha=0.5)
plt.scatter(
    range(len(y_train)),
    regressor.predict(X_train),
    color="blue",
    label="Predicted",
    alpha=0.5,
)
plt.title("Real vs Predicted (Train)")
plt.xlabel("Sample index")
plt.ylabel("Target")
plt.legend()

# Test
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test.values, color="red", label="Real", alpha=0.5)
plt.scatter(range(len(y_test)), y_pred, color="blue", label="Predicted", alpha=0.5)
plt.title("Real vs Predicted (Test)")
plt.xlabel("Sample index")
plt.ylabel("Target")
plt.legend()
plt.show()
