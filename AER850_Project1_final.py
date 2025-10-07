
# AER850 Project 1 

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

os.chdir(Path(__file__).parent)

CSV_PATH = "Project 1 Data.csv"
RANDOM_STATE = 42
OUT_MODEL = "best_model.joblib"
OUT_METRICS = "metrics.txt"

# 2.1 — Data Processing
print("=== Step 2.1: Data Processing ===")
df = pd.read_csv(CSV_PATH)
X = df[["X","Y","Z"]].values
y = df["Step"].values
print("shape:", df.shape)
print("class counts:\n", df["Step"].value_counts().sort_index())

# 2.2 — Data Visualization
print("\n=== Step 2.2: Data Visualization ===")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col in zip(axes, ["X","Y","Z"]):
    ax.hist(df[col], bins=30, edgecolor="black", alpha=0.75)
    ax.axvline(df[col].mean(), linestyle="--", linewidth=1)
    ax.set_title(col); ax.set_xlabel(col); ax.set_ylabel("Count"); ax.grid(True, linestyle=":", linewidth=0.5)
fig.suptitle("Feature Distributions", y=1.05)
plt.tight_layout(); plt.show()

steps_sorted = sorted(df.Step.unique())
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True)
for ax, col in zip(axes, ["X","Y","Z"]):
    data = [df.loc[df.Step==c, col] for c in steps_sorted]
    ax.boxplot(data, showfliers=False)
    ax.set_title(f"{col} by Step"); ax.set_xlabel("Step"); ax.set_ylabel(col)
    ax.set_xticks(range(1, len(steps_sorted)+1)); ax.set_xticklabels(steps_sorted)
    ax.grid(True, linestyle=":", linewidth=0.5)
plt.tight_layout(); plt.show()

# 2.3 — Correlation Analysis (Pearson)
print("\n=== Step 2.3: Correlation Analysis (Pearson) ===")
corr_cols = ["X","Y","Z","Step"]
corr = df[corr_cols].corr(method="pearson")
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="coolwarm")
plt.title("Pearson correlation (X, Y, Z, Step)")
plt.tight_layout(); plt.show()

# 2.4 — Classification Model Development
print("\n=== Step 2.4: Classification Model Development ===")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = []
logreg = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
logreg_grid = {"clf__C": [0.1, 1.0, 10.0]}; models.append(("LogisticRegression", logreg, logreg_grid, "grid"))
knn = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
knn_grid = {"clf__n_neighbors": [3,5,7], "clf__weights": ["uniform","distance"]}; models.append(("KNN", knn, knn_grid, "grid"))
dt = Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
dt_grid = {"clf__max_depth": [None, 5, 10], "clf__min_samples_split": [2,5,10]}; models.append(("DecisionTree", dt, dt_grid, "grid"))
rf = Pipeline([("clf", RandomForestClassifier(random_state=RANDOM_STATE))])
rf_dist = {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [None, 5, 10, 20], "clf__max_features": ["sqrt", "log2", None], "clf__min_samples_split": [2, 5, 10]}
models.append(("RandomForest", rf, rf_dist, "random"))

results = []
best_clf = None
best_name = None
best_cv = -np.inf

for name, pipe, space, mode in models:
    if mode == "grid":
        search = GridSearchCV(pipe, space, cv=cv, n_jobs=-1, scoring="f1_macro", refit=True)
    else:
        search = RandomizedSearchCV(pipe, space, n_iter=20, cv=cv, n_jobs=-1, scoring="f1_macro", random_state=RANDOM_STATE, refit=True)
    search.fit(X_tr, y_tr)
    cv_score = search.best_score_
    y_pred_tmp = search.predict(X_te)
    test_f1 = f1_score(y_te, y_pred_tmp, average="macro")
    test_acc = accuracy_score(y_te, y_pred_tmp)
    test_prec = precision_score(y_te, y_pred_tmp, average="macro", zero_division=0)
    results.append((name, cv_score, test_f1, test_prec, test_acc, search.best_params_))
    if cv_score > best_cv:
        best_cv, best_clf, best_name = cv_score, search.best_estimator_, name

# 2.5 — Model Performance Analysis
print("\n=== Step 2.5: Model Performance Analysis ===")
print("Model | CV F1 (macro) | Test F1 (macro) | Test Precision (macro) | Test Accuracy")
for row in results:
    print(f"{row[0]:<16} {row[1]:7.3f} {row[2]:14.3f} {row[3]:22.3f} {row[4]:14.3f}")

y_pred = best_clf.predict(X_te)
report = classification_report(y_te, y_pred, digits=3)
cm = confusion_matrix(y_te, y_pred)
print("\nSelected model:", best_name)
print(report)

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(np.unique(y)))
disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
ax.set_title(f"Confusion Matrix — {best_name}")
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
plt.tight_layout(); plt.show()

# 2.6 — Stacked Model Performance Analysis
print("\n=== Step 2.6: Stacked Model Performance Analysis ===")
base1 = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
base2 = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))])
stack = StackingClassifier(estimators=[("logreg", base1), ("knn", base2)],
                           final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                           n_jobs=-1)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
stack.fit(X_tr2, y_tr2)
y_pred_stack = stack.predict(X_te2)
print("Stack macro-F1:", f1_score(y_te2, y_pred_stack, average="macro"))
print("Stack precision (macro):", precision_score(y_te2, y_pred_stack, average="macro", zero_division=0))
print("Stack accuracy:", accuracy_score(y_te2, y_pred_stack))
print("Stack report:\n", classification_report(y_te2, y_pred_stack, digits=3))
print("Stack confusion matrix:\n", confusion_matrix(y_te2, y_pred_stack))

# 2.7 — Model Evaluation
print("\n=== Step 2.7: Model Evaluation (Predict 5 points) ===")
joblib.dump(best_clf, OUT_MODEL)

with open(OUT_METRICS, "w") as f:
    f.write("Model comparison (CV F1 macro, Test F1, Test Precision macro, Test Acc)\n")
    for row in results:
        f.write(f"{row[0]:<17} | CV F1: {row[1]:.3f} | Test F1: {row[2]:.3f} | Test Prec(m): {row[3]:.3f} | Test Acc: {row[4]:.3f} | Best Params: {row[5]}\n")
    f.write(f"\nSelected model: {best_name}\n\n")
    f.write("Classification report:\n"); f.write(report)
    f.write("\nConfusion matrix:\n"); f.write(np.array2string(cm))

points_to_predict = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3],
]
preds = best_clf.predict(np.array(points_to_predict, dtype=float)).tolist()
print("Predictions for the 5 points:")
for pt, p in zip(points_to_predict, preds):
    print(f"{pt} -> Step {p}")
    
    # Final Product
