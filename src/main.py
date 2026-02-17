import pandas as pd
from matplotlib import pyplot as plt

DATA_FILE = "../data/drugLibTest_raw.tsv"

data = pd.read_csv(DATA_FILE, sep="\t", engine="python")

print("Общая информация о dataset:")
data.info()
print("\n")

print("Первые 5 строк:\n", data.head(), "\n")
print("Последние 5 строк:\n", data.tail(), "\n")

print("\nСтатистика по числовым колонкам:\n", data.describe(), "\n")
print("\nСтатистика по колонкам с текстом:\n", data["urlDrugName"].describe(), "\n")
print("\nСтатистика по колонкам с текстом:\n", data["effectiveness"].describe(), "\n")
print("\nСтатистика по колонкам с текстом:\n", data["sideEffects"].describe(), "\n")
print("\nСтатистика по колонкам с текстом:\n", data["condition"].describe(), "\n")

print("Кол-во строк, столбцов: ", data.shape, "\n")
print("Имена столбцов:\n", data.columns, "\n")

data["benefits_len"] = data["benefitsReview"].str.len()
data["sideEffects_len"] = data["sideEffectsReview"].str.len()
data["comments_len"] = data["commentsReview"].str.len()

print(
    "Корреляция от длины benefits и RATING:\n", data[["benefits_len", "rating"]].corr()
)
print(
    "\nКорреляция от длины sideEffects и RATING:\n",
    data[["sideEffects_len", "rating"]].corr(),
)
print(
    "\nКорреляция от длины comments и RATING:\n",
    data[["comments_len", "rating"]].corr(),
)

plt.figure(1)
plt.subplot(3, 1, 1)
plt.scatter(data["benefits_len"], data["rating"])
plt.xlabel("Benefits Len")
plt.ylabel("Rating")
plt.subplot(3, 1, 2)
plt.scatter(data["sideEffects_len"], data["rating"])
plt.xlabel("SideEffects Len")
plt.ylabel("Rating")
plt.subplot(3, 1, 3)
plt.scatter(data["comments_len"], data["rating"])
plt.xlabel("Comments Len")
plt.ylabel("Rating")

plt.tight_layout()
plt.show()
