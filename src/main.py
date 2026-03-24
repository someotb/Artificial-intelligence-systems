import numpy as np
import pandas as pd
import seaborn as sns
import func
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error
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
class_names = [str(c) for c in sorted(y.unique())]

# Значения выбраны пользователем
test_tree = DecisionTreeClassifier(max_depth=7, random_state=42, max_features=10)
test_tree.fit(X_train, y_train)
tree.export_graphviz( 
    test_tree, 
    feature_names=X.columns, 
    class_names=class_names,
    out_file="drug_tree_test.dot", 
    filled=True, 
    rounded=True
)

# Найдем для этого тестового дерева оптимальные значения путем подстановки и сравнения
depth_list = range(1,11)
features_list = range(1,11)
mean_list = []
best_depth = 0
best_feature = 0
best_mean = 0

for depth in depth_list:
    for feature in features_list:
        test_tree_range = DecisionTreeClassifier(max_depth=depth, random_state=42, max_features=feature)
        cros_val = cross_val_score(test_tree_range, X, y, cv=10)
        mean = cros_val.mean()
        mean_list.append(mean)
        if mean > best_mean:
            best_mean = mean
            best_depth = depth
            best_feature = feature
        
print("Лучшие значения:")
print(f"Depth = {best_depth}")
print(f"Feature = {best_feature}")
print(f"Mean = {best_mean}")

plt.figure(1, figsize=(10, 6))
plt.plot(mean_list)
plt.axvline(x=mean_list.index(best_mean), color='r', linestyle='--', label='Best score')
plt.xlabel("Deapth(i) + Feature(i)")
plt.ylabel("Mean cross validation score")
plt.legend()

# Обучения с автоматическим подбором оптимальных значений через `GridSearchCV`
dtc = DecisionTreeClassifier(max_depth=4, random_state=42, max_features=7)
tree_params = { "max_depth": range(1,20), "max_features": range(1,10) }
tree_grid = GridSearchCV(dtc, tree_params, cv=10, verbose=True, n_jobs=-1)
tree_grid.fit(X_train, y_train)

print("\n")
print(f"Лучшее сочетание параметров: {tree_grid.best_params_}")
print(f"Лучшее баллы cross val: {tree_grid.best_score_}")


# Полученное дерево решений можно посмотреть можно тут: http://webgraphviz.com/
tree.export_graphviz( 
    tree_grid.best_estimator_, 
    feature_names=X.columns, 
    class_names=class_names,
    out_file="drug_tree.dot", 
    filled=True, 
    rounded=True
)

best_tree = tree_grid.best_estimator_
predict = best_tree.predict(X_holdout)
accur = accuracy_score(y_holdout, predict)
mae = mean_absolute_error(y_holdout, predict)

importances = pd.Series(best_tree.feature_importances_, index=X.columns)
print("Важность признаков:\n", importances.sort_values(ascending=False))

print(f"Точность: {accur}")
print(f"Mean absolute error: {mae}")

y_test_values = y_holdout.values 
print(f"\nOriginal | Predict")
for i in range(10):
    print(f"{y_test_values[i]} | {predict[i]}")
    
# Построим график решающих границ
# Сперва выделим два признака, по которым мы будем строить график
features_boundaries = ["effectiveness", "sideEffects"]
X_vis = X[features_boundaries]
y_vis = y

model_vis = DecisionTreeClassifier(random_state=42, max_depth=4)
model_vis.fit(X_vis, y_vis)

disp = DecisionBoundaryDisplay.from_estimator(
    model_vis, 
    X_vis, 
    response_method="predict",
    cmap=plt.cm.viridis,
    alpha=0.5,
    xlabel="Эффективность",
    ylabel="Побочные эффекты"
)

plt.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=y_vis, edgecolor="k", s=20, cmap=plt.cm.RdYlGn)
plt.title("Решающие границы дерева решений (на 2-х признаках)")
    
plt.show()
