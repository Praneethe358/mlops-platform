import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("customer_churn_prediction_dataset.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
X = df.drop("Churn", axis=1)
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)