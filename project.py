import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load CSV data
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/data.csv")

# Split features and labels
X = data.drop("label", axis=1)
y = data["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Predictions:", y_pred)
print("Test Accuracy:", acc)

pickle.dump(model, open("/content/drive/MyDrive/Colab Notebooks/model.pkl", "wb"))