import numpy as np
import pandas as pd
import seaborn as sns
import func
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_FILE = "../data/drugLibTest_raw.tsv"

data = pd.read_csv(DATA_FILE, sep="\t", engine="python")

data["benefitsReview"] = data["benefitsReview"].fillna("")
data["sideEffectsReview"] = data["sideEffectsReview"].fillna("")
data["commentsReview"] = data["commentsReview"].fillna("")

func.check_for_unic(data)

# Переводим str в int для обучения, сохраняя порядок, где больше = лучше
eff_map = {
    'Ineffective': 0,
    'Marginally Effective': 1,
    'Moderately Effective': 2,
    'Considerably Effective': 3,
    'Highly Effective': 4
}

side_map = {
    'No Side Effects': 0,
    'Mild Side Effects': 1,
    'Moderate Side Effects': 2,
    'Severe Side Effects': 3,
    'Extremely Severe Side Effects': 4
}

data['effectiveness'] = data['effectiveness'].map(eff_map)
data['sideEffects'] = data['sideEffects'].map(side_map)

# Получаем тон комментариев с помощью TextBlob
data["benefits_tone"] = data["benefitsReview"].apply(func.get_tone)
data["sideEffects_tone"] = data["sideEffectsReview"].apply(func.get_tone)
data["comments_tone"] = data["commentsReview"].apply(func.get_tone)

# Заменяем название болезни на средний рейтинг по этой болезни 
le = LabelEncoder()
data["condition"] = le.fit_transform(data["condition"].astype(str))
data["urlDrugName"] = le.fit_transform(data["urlDrugName"].astype(str))

X = data[["urlDrugName", "effectiveness", "sideEffects", "condition", "benefits_tone", "sideEffects_tone", "comments_tone"]]
y = data["rating"]

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

dtc = DecisionTreeClassifier(max_depth=4, random_state=42, max_features=7)
tree_params = { "max_depth": range(1,20), "max_features": range(1,10) }
tree_grid = GridSearchCV(dtc, tree_params, cv=10, verbose=True, n_jobs=-1)
tree_grid.fit(X_train, y_train)

print("\n")
print(f"Лучшее сочетание параметров: {tree_grid.best_params_}")
print(f"Лучшее баллы cross val: {tree_grid.best_score_}")

class_names = [str(c) for c in sorted(y.unique())]

# Полученное дерево решений можно посмотреть можно тут: http://webgraphviz.com/
tree.export_graphviz( 
    tree_grid.best_estimator_, 
    feature_names=X.columns, 
    class_names=class_names,
    out_file="drug_tree.dot", 
    filled=True, 
    rounded=True
)
# Само дерево
plt.figure(figsize=(20, 10))
plot_tree(tree_grid.best_estimator_, 
          feature_names=X.columns, 
          class_names=class_names, 
          filled=True, 
          fontsize=10)
plt.savefig('tree.png')

best_tree = tree_grid.best_estimator_
predict = best_tree.predict(X_holdout)
accur = accuracy_score(y_holdout, predict)
mae = mean_absolute_error(y_holdout, predict)

print(f"Точность: {accur}")
print(f"Mean absolute error: {mae}")

y_test_values = y_holdout.values 
print(f"\nOriginal | Predict")
for i in range(10):
    print(f"{y_test_values[i]} | {predict[i]}")