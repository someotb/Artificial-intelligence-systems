import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, median_absolute_error, explained_variance_score

warnings.filterwarnings("ignore")

# Загрузка
X_train = pd.read_csv("data/lab6_X_train.csv")
X_test  = pd.read_csv("data/lab6_X_test.csv")
y_train = pd.read_csv("data/lab6_Y_train.csv").values.ravel()
y_test  = pd.read_csv("data/lab6_Y_test.csv").values.ravel()

# Берём ОДИН признак — effectiveness
X_train_1d = X_train[["effectiveness"]].values
X_test_1d  = X_test[["effectiveness"]].values

# Полиномиальные признаки
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train_1d)
X_test_poly  = poly.transform(X_test_1d)

# Обучение
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Предсказание и метрики
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, median_absolute_error, explained_variance_score

y_pred = model.predict(X_test_poly)

mae     = mean_absolute_error(y_test, y_pred)
mse     = mean_squared_error(y_test, y_pred)
rmse    = np.sqrt(mse)
r2      = r2_score(y_test, y_pred)
med_ae  = median_absolute_error(y_test, y_pred)

print(f"{'МЕТРИКИ МОДЕЛИ':^35}")
print(f"MAE   (ср. ошибка):       {mae:.4f}")
print(f"MedAE (медианная ошибка): {med_ae:.4f}")
print(f"MSE   (ср. кв. ошибка):   {mse:.4f}")
print(f"RMSE  (корень из MSE):     {rmse:.4f}")
print(f"R²    (детерминация):      {r2:.4f}")
print("\n")
print(f"Среднее y_test:   {np.mean(y_test):.4f}")
print(f"Std y_test:       {np.std(y_test):.4f}")
print(f"Min предсказание: {y_pred.min():.4f}")
print(f"Max предсказание: {y_pred.max():.4f}")

X_grid = np.arange(X_train_1d.min(), X_train_1d.max() + 0.1, 0.1).reshape(-1, 1)
y_grid = model.predict(poly.transform(X_grid))

# График
plt.figure(figsize=(9, 5))
plt.scatter(X_test_1d, y_test, color="red", label="Реальные значения", zorder=5)
plt.plot(X_grid, y_grid, color="blue", linewidth=2, label="Poly degree=4")
plt.xlabel("Effectiveness")
plt.ylabel("Rating")
plt.title("Polynomial Regression — Effectiveness vs Rating")
plt.legend()
plt.tight_layout()
plt.show()