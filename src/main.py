import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as skl
from matplotlib import pyplot as plt

DATA_FILE = "../data/drugLibTest_raw.tsv"

data = pd.read_csv(DATA_FILE, sep="\t", engine="python")

print("Общая информация о dataset:")
data.info()
print("\n")

print("Кол-во строк, столбцов: ", data.shape, "\n")
print("Имена столбцов:\n", data.columns, "\n")

data["benefits_len"] = data["benefitsReview"].str.len().fillna(0).astype(int)
data["sideEffects_len"] = data["sideEffectsReview"].str.len().fillna(0).astype(int)
data["comments_len"] = data["commentsReview"].str.len().fillna(0).astype(int)

int_data = data[["rating", "benefits_len", "sideEffects_len", "comments_len"]]
str_data = data[
    [
        "urlDrugName",
        "effectiveness",
        "sideEffects",
        "condition",
        "benefitsReview",
        "sideEffectsReview",
        "commentsReview",
    ]
]

# PairPlot
sns.pairplot(int_data, hue="rating")

# Классификация
X, X_holdout, y, y_holdout = skl.model_selection.train_test_split(
    int_data[["benefits_len", "sideEffects_len", "comments_len"]],
    int_data[["rating"]].values.ravel(),
    test_size=0.2,
    random_state=42,
)

K_list = np.arange(1, 51)
scores_list = []

skf = skl.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in K_list:
    knn_test = skl.neighbors.KNeighborsClassifier(n_neighbors=k)
    scores = skl.model_selection.cross_val_score(knn_test, X, y, cv=skf, scoring="accuracy")
    scores_list.append(scores.mean())

ER = [1 - x for x in scores_list]

plt.figure(2)
plt.plot(K_list, ER)
plt.xlabel("Кол-во соседей")
plt.ylabel("Ошибка классификации(ER)")

best_K = []
min_ER = min(ER)
for i in range(len(ER)):
    if ER[i] <= min_ER:
        best_K.append(i)

print(f"Оптимальные значения K:")
for k in best_K:
    print(f"Index: {k} | ER[{k}]: {ER[k]}")

knn = skl.neighbors.KNeighborsClassifier(n_neighbors=best_K[0])
knn.fit(X, y)
knn_predict = knn.predict(X_holdout)
accuracy = skl.metrics.accuracy_score(y_holdout, knn_predict)

print("\nТаблица сравнения Predict и Original:")
for i in range(len(knn_predict)):
    if (i > 0 and i < 10):
        if (knn_predict[i] == y_holdout[i]):
            print(f"\tPredict: {knn_predict[i]}, Original: {y_holdout[i]} | Hit")
        else:
            print(f"\tPredict: {knn_predict[i]}, Original: {y_holdout[i]} | Miss")

print(f"\nAccuracy: {accuracy}")

plt.tight_layout()
plt.show()
