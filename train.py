# train.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# 1. Load Dataset
df = pd.read_csv("smartdemand.csv")

# 2. Define Features & Target
X = df[["month", "store_id", "product_id", "marketing_spend", "discount_rate"]]
y = df["sales"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save Model and Test Set
joblib.dump(model, "smartdemand_rf_model.pkl")
X_test.to_csv("test_features.csv", index=False)
y_test.to_csv("test_labels.csv", index=False)

print("✅ Model trained and saved as 'smartdemand_rf_model.pkl'")
print("✅ Test features and labels saved as 'test_features.csv' and 'test_labels.csv'")
