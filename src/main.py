import pandas as pd

DATA_FILE = "../data/drugLibTest_raw.tsv"

data = pd.read_csv(DATA_FILE, sep="\t", engine="python")

print("Общая информация о dataset:")
data.info()
print("\n")


print("Первые 5 строк:\n", data.head(), "\n")
print("Последние 5 строк:\n", data.tail(), "\n")

print("\nСтатистика по числовым колонкам:")
print(data.describe(), "\n")

print("Кол-во строк, столбцов: ", data.shape, "\n")
print("Имена столбцов:\n", data.columns, "\n")

# print(data.iloc[0])
