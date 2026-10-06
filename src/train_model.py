import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

DATA = "data/student_performance_demo.csv"

df = pd.read_csv(DATA)

print("Dataset Shape:", df.shape)
print("\nDataset Columns:")
print(df.columns.tolist())

print("\nPerformance Distribution:")
print(df["performance"].value_counts())


# -----------------------------------
# 2. Features and Target
# -----------------------------------

X = df.drop(columns=["performance"])
y = df["performance"]


# -----------------------------------
# 3. Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 4. Define Models
# -----------------------------------

models = {

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=4
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])
}


# -----------------------------------
# 5. Train and Compare Models
# -----------------------------------

results = {}

for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    print(
        "Test Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


# -----------------------------------
# 6. Best Model
# -----------------------------------

best_model_name = max(
    results,
    key=results.get
)

best_model = models[best_model_name]

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

for name, accuracy in results.items():

    print(
        name,
        ":",
        round(accuracy * 100, 2),
        "%"
    )


print("\nBest Model:", best_model_name)
print(
    "Best Accuracy:",
    round(results[best_model_name] * 100, 2),
    "%"
)


# -----------------------------------
# 7. Train Best Model Again
# -----------------------------------

best_model.fit(X_train, y_train)

final_predictions = best_model.predict(X_test)


# -----------------------------------
# 8. Confusion Matrix
# -----------------------------------

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        final_predictions,
        labels=["Low", "Medium", "High"]
    )
)


# -----------------------------------
# 9. Feature Importance
# -----------------------------------

if best_model_name == "Random Forest":

    importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(
        ascending=False
    )

    print("\nFeature Importance:")
    print(importance)


# -----------------------------------
# 10. Save Final Model
# -----------------------------------

joblib.dump(
    best_model,
    "student_performance_model.pkl"
)

print(
    "\nFinal model saved as "
    "student_performance_model.pkl"
)