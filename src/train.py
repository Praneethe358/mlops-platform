import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv("data/churn.csv")

# Remove useless ID column
df = df.drop("customerID", axis=1)

# Encode target
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.4f}")


print(classification_report(
    y_test,
    predictions
))

# Save model
joblib.dump(model, "models/model.pkl")
print(df["Churn"].value_counts())
print("Model saved successfully!")