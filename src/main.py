import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.core.methods.selectn import DataFrame

DATA_FILE = "../data/drugLibTest_raw.tsv"

data = pd.read_csv(DATA_FILE, sep="\t", engine="python")

print("Общая информация о dataset:")
data.info()
print("\n")

print("Кол-во строк, столбцов: ", data.shape, "\n")
print("Имена столбцов:\n", data.columns, "\n")

data["benefits_len"] = data["benefitsReview"].str.len()
data["sideEffects_len"] = data["sideEffectsReview"].str.len()
data["comments_len"] = data["commentsReview"].str.len()

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

drugs = [d for d in data.columns if "paxil" in d]
effectivnes = [e for e in data.columns if "Highly Effective" in e]
side_effect = [d for d in data.columns if "Mild Side Effects" in d]
condition = [d for d in data.columns if "Depression" in d]

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

plt.figure(2)
plt.title("Correlation Heatmap")
sns.heatmap(int_data.corr(), cmap="coolwarm", center=0)

plt.figure(3)

plt.subplot(1, 4, 1)
plt.xlabel("Рейтинг")
plt.ylabel("Кол-во оценок")
data["rating"].hist(figsize=(10, 5))

plt.subplot(1, 4, 2)
plt.xlabel("Длина отзыва")
plt.ylabel("Кол-во отзывов")
data["benefits_len"].hist(figsize=(10, 5))

plt.subplot(1, 4, 3)
plt.xlabel("Длина отзыва")
plt.ylabel("Кол-во отзывов")
data["sideEffects_len"].hist(figsize=(10, 5))

plt.subplot(1, 4, 4)
plt.xlabel("Длина отзыва")
plt.ylabel("Кол-во отзывов")
data["comments_len"].hist(figsize=(10, 5))

# plt.figure(4)
# sns.pairplot(str_data[drugs])
# plt.subplot(2, 2, 1)
# plt.xlabel("Рейтинг")
# plt.ylabel("Кол-во оценок")
# data[drugs].hist(figsize=(5, 5))

# plt.subplot(2, 2, 2)
# plt.xlabel("Длина отзыва")
# plt.ylabel("Кол-во отзывов")
# data[effectivnes].hist(figsize=(5, 5))

# plt.subplot(2, 2, 3)
# plt.xlabel("Длина отзыва")
# plt.ylabel("Кол-во отзывов")
# data[side_effect].hist(figsize=(5, 5))

# plt.subplot(2, 2, 4)
# plt.xlabel("Длина отзыва")
# plt.ylabel("Кол-во отзывов")
# data[condition].hist(figsize=(5, 5))

plt.tight_layout()
plt.show()
