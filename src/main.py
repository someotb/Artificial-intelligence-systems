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

min_mae = 100
min_med = 100
best_degree = []
degrees = []
maes = []

for deg in range(1, 21):
    # Полиномиальные признаки
    poly_test = PolynomialFeatures(degree=deg)
    X_train_poly_t = poly_test.fit_transform(X_train_1d)
    X_test_poly_t  = poly_test.transform(X_test_1d)

    # Обучение
    model_t = LinearRegression()
    model_t.fit(X_train_poly_t, y_train)

    # Предсказание и метрики
    y_pred = model_t.predict(X_test_poly_t)
    mae     = mean_absolute_error(y_test, y_pred)

    degrees.append(deg)
    maes.append(mae)

    if min_mae > mae:
        min_mae = mae
        best_degree.append(deg)

# Полиномиальные признаки
poly = PolynomialFeatures(degree=best_degree[-1])
X_train_poly = poly.fit_transform(X_train_1d)
X_test_poly  = poly.transform(X_test_1d)

# Обучение
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Предсказание и метрики
y_pred = model.predict(X_test_poly)

print(f"Лучшее значение MAE: {min_mae} для degree = {best_degree[-1]}")
print(f"Все значения degree: {[deg for deg in best_degree]}\n")

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
plt.plot(X_grid, y_grid, color="blue", linewidth=2, label=f"Poly degree={best_degree[-1]}")
plt.xlabel("Effectiveness")
plt.ylabel("Rating")
plt.title("Polynomial Regression — Effectiveness vs Rating")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(degrees, maes, label="Зависимость MAE от degrees")
plt.xlabel("Degrees values")
plt.ylabel("MAE")
plt.title("Зависимость MAE от degrees")
plt.legend()

plt.show()