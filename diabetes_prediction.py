
# CODEALPHA INTERNSHIP - TASK 4
# DIABETES PREDICTION USING MACHINE LEARNING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# STEP 1: LOAD DATASET

print("Loading Diabetes Dataset...")

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

df = pd.read_csv(url)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# STEP 2: SEPARATE FEATURES AND TARGET

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# In this dataset, zero in selected measurements
# represents a missing or invalid measurement.
zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[zero_columns] = X[zero_columns].replace(0, np.nan)

print("\nMissing values:")
print(X.isnull().sum())

# STEP 3: SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 4: CREATE MACHINE LEARNING MODELS

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=150,
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ])
}

# STEP 5: TRAIN AND EVALUATE MODELS

results = {}

print("\nTRAINING MODELS...\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print("\n", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

# STEP 6: SELECT BEST MODEL

best_name = max(results, key=results.get)
best_model = models[best_name]

print("\nBest model:", best_name)

# STEP 7: SAVE MODEL

joblib.dump(best_model, "diabetes_model.pkl")

print("Model saved successfully!")

# STEP 8: MODEL COMPARISON GRAPH

plt.figure(figsize=(9, 5))

plt.bar(
    results.keys(),
    [value * 100 for value in results.values()],
    color=["skyblue", "orange", "green"]
)

plt.title("Diabetes Prediction - Model Comparison")
plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Accuracy (%)")
plt.tight_layout()

plt.savefig("diabetes_model_comparison.png")
plt.show()

# STEP 9: CONFUSION MATRIX

predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diabetes", "Diabetes"]
)

display.plot(cmap="Blues")
plt.title("Diabetes Prediction Confusion Matrix")
plt.tight_layout()

plt.savefig("diabetes_confusion_matrix.png")
plt.show()

print("\nDIABETES PREDICTION MODEL COMPLETED!")
