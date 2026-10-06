import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/student_performance_demo.csv")

# Features and target
X = df.drop("performance", axis=1)
y = df["performance"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# Final Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

# Train model
model.fit(X_train_scaled, y_train)

# Feature coefficients
coefficients = pd.DataFrame(
    model.coef_,
    columns=X.columns,
    index=model.classes_
)

print("\nFeature Coefficients:")
print(coefficients)

# Average absolute importance
feature_importance = coefficients.abs().mean().sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot
plt.figure(figsize=(10, 6))

feature_importance.sort_values().plot(
    kind="barh"
)

plt.title("Feature Importance - Logistic Regression")
plt.xlabel("Average Absolute Coefficient")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.show()