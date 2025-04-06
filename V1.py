import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Generate synthetic data (same as original)
np.random.seed(42)
n_days = 365
dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
temperature = 10 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 3, n_days)
consumption = 50000 - 800 * temperature + np.random.normal(0, 2000, n_days)

df = pd.DataFrame({
    'Date': dates,
    'Temperature': temperature,
    'Consumption': consumption
})

# 2. Feature engineering
df['DayOfYear'] = df['Date'].dt.dayofyear
X = df[['Temperature', 'DayOfYear']]
y = df['Consumption']

# 3. Time-based train/test split (80/20)
split_idx = int(0.8 * len(df))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_train, dates_test = dates[:split_idx], dates[split_idx:]

# 4. Model training
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# 6. Evaluation metrics
def print_metrics(y_true, y_pred, label):
    print(f"\n{label} Metrics:")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")

print_metrics(y_train, train_pred, "Training")
print_metrics(y_test, test_pred, "Testing")

# 7. Visualization with improved visibility
plt.figure(figsize=(14, 6))

# Training data - using thicker lines and distinct colors
plt.plot(dates_train, y_train, 'navy', linewidth=2.5, label='Actual (Train)')
plt.plot(dates_train, train_pred, 'royalblue', linestyle='--', linewidth=2, label='Predicted (Train)')

# Testing data - using high-contrast colors
plt.plot(dates_test, y_test, 'darkred', linewidth=2.5, label='Actual (Test)')
plt.plot(dates_test, test_pred, 'red', linestyle='--', linewidth=2, label='Predicted (Test)')

plt.title('Electricity Consumption: Actual vs Predicted', fontsize=14, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Consumption (MW)', fontsize=12)
plt.legend(fontsize=12, framealpha=1)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add vertical line to show train/test split
split_date = dates[split_idx]
plt.axvline(x=split_date, color='black', linestyle=':', linewidth=1.5)
plt.text(split_date, plt.ylim()[1]*0.95, ' Train/Test Split ', 
         ha='right', va='top', backgroundcolor='white')

plt.show()

# 8. Residual analysis with improved visibility
residuals = y_test - test_pred
plt.figure(figsize=(12, 5))
plt.scatter(test_pred, residuals, c='darkgreen', alpha=0.7, s=60, edgecolor='white')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
plt.title('Residual Plot (Test Set)', fontsize=14, pad=15)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()