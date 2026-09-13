import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("combined_dataset.csv")

X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data ready:", X_scaled.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

print("Random Forest model trained successfully!")
import joblib

joblib.dump({
    "model": model,
    "scaler": scaler,
    "features": ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
    "classes": list(model.classes_)
}, "crop_model.joblib")

print("Model saved successfully as crop_model.joblib!")
import joblib

joblib.dump({
    "model": model,
    "scaler": scaler,
    "features": ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
    "classes": list(model.classes_)
}, "crop_model.joblib")

print("Model saved successfully as crop_model.joblib!")