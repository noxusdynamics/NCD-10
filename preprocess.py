import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("combined_dataset.csv")  # change filename if needed

print(data.isnull().sum())
print("Duplicate rows:", data.duplicated().sum())

data = data.dropna()
data = data.drop_duplicates()

X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Final shape:", X.shape)
print(y.value_counts())
