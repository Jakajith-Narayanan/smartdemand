# SmartDemand

SmartDemand is a sales forecasting project that uses a **Random Forest Regressor** to predict product sales based on historical data including month, store, product details, marketing spend, and discount rates.

---

## Features

- Uses a CSV dataset (`smartdemand.csv`) containing your business’s sales data
- Trains a Random Forest model to predict sales based on your data
- Allows input-based prediction on new data
- Simple and extendable for various business scenarios

---

## Project Structure

- `generate_dataset.py` — Generates a sample synthetic dataset (`smartdemand.csv`) with 250 samples for demo/testing purposes
- `train.py` — Loads your dataset (`smartdemand.csv`), trains the Random Forest model, and saves it (`model.joblib`)
- `test.py` — Loads saved model and predicts sales based on user inputs
- `smartdemand.csv` — The main dataset file used for training (replace with your own business data)
- `model.joblib` — Trained model file

---

## Important: Using This Project for Your Business

To apply SmartDemand to your own business:

1. Replace the `smartdemand_dataset.csv` file with your own sales data CSV.  
   - Make sure the CSV columns correspond to the expected features:
     - `month` (int): Month number (1-12)  
     - `store_id` (int): Store identifier  
     - `product_id` (int): Product identifier  
     - `marketing_spend` (float): Marketing budget spent  
     - `discount_rate` (float): Discount applied (0 to 1)  
     - `sales` (float): Actual sales amount (target variable)

2. Adjust the column names in `train.py` if your CSV has different column names.

3. Run the training and testing scripts as usual.

---

## Requirements

- Python 3.x
- pandas
- scikit-learn
- joblib
- numpy

Install dependencies with:

```bash
pip install pandas scikit-learn joblib numpy
