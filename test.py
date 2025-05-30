# test.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the trained model
model = joblib.load("smartdemand_rf_model.pkl")
print("✅ Model loaded successfully.")

# 2. Live input from user
def get_user_input():
    print("\n🔍 Enter values to predict sales:")
    month = int(input("Enter month (1–12): "))
    store_id = int(input("Enter store ID (e.g., 101, 102, 103): "))
    product_id = int(input("Enter product ID (e.g., 501–504): "))
    marketing_spend = int(input("Enter marketing spend (₹): "))
    discount_rate = float(input("Enter discount rate (%): "))

    input_data = pd.DataFrame([{
        "month": month,
        "store_id": store_id,
        "product_id": product_id,
        "marketing_spend": marketing_spend,
        "discount_rate": discount_rate
    }])
    return input_data

user_input = get_user_input()

# 3. Predict on user input
predicted_sales = model.predict(user_input)[0]
print(f"\n📈 Predicted Sales: ₹{predicted_sales:.2f}")

# 4. Load saved test set
X_test = pd.read_csv("test_features.csv")
y_test = pd.read_csv("test_labels.csv").squeeze()

# 5. Predict and evaluate on saved test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n📊 Model Evaluation on Test Set:")
print(f"   - Mean Squared Error: {mse:.2f}")
print(f"   - R² Score: {r2:.2f}")

# 6. Plot sample predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:50], label="Actual Sales", marker='o')
plt.plot(y_pred[:50], label="Predicted Sales", linestyle='--', marker='x')
plt.title("SmartDemand: Actual vs Predicted Sales (Top 50 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
