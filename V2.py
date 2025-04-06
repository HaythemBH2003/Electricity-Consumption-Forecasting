import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Generate synthetic data with clear yearly seasonality
np.random.seed(42)
n_days = 365 * 3  
dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')

# Base temperature with yearly seasonality
temperature = 10 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 3, n_days)

# Consumption in MW with winter/summer peaks
consumption = (
    50000 
    - 800 * temperature 
    + 300 * np.sin(2 * np.pi * dates.dayofyear / 365)  # Yearly pattern
    + np.random.normal(0, 2000, n_days)
)

df = pd.DataFrame({
    'Date': dates,
    'Temperature': temperature,
    'Consumption': consumption
})

# 2. Resample to ensure daily frequency (in case of gaps)
df = df.set_index('Date').asfreq('D').reset_index()

# 3. Time series diagnostics
def plot_diagnostics(df):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Raw data
    df.plot(x='Date', y='Consumption', ax=axes[0], title='Raw Consumption Data')
    
    # Autocorrelation
    plot_acf(df['Consumption'], lags=365, ax=axes[1], title='Autocorrelation (1 year)')
    

plot_diagnostics(df)

# 4. Proper seasonal decomposition
def decompose_timeseries(df, period=365):
    try:
        stl = STL(
            df.set_index('Date')['Consumption'],
            period=period,
            seasonal=13  
        )
        res = stl.fit()
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
        res.observed.plot(ax=ax1, title='Observed')
        res.trend.plot(ax=ax2, title='Trend')
        res.seasonal.plot(ax=ax3, title=f'Seasonal (period={period})')
        res.resid.plot(ax=ax4, title='Residuals')
        plt.tight_layout()
        plt.show()
        
        return res
    except Exception as e:
        print(f"Decomposition failed: {e}")
        return None

# First try yearly decomposition
yearly_decomp = decompose_timeseries(df, period=365)

# If still seeing monthly patterns, try weekly decomposition
if yearly_decomp and abs(yearly_decomp.seasonal).max() < 1000:  
    weekly_decomp = decompose_timeseries(df, period=7)

# 5. Feature engineering based on decomposition results
df['DayOfYear'] = df['Date'].dt.dayofyear
df['WeekOfYear'] = df['Date'].dt.isocalendar().week
df['Month'] = df['Date'].dt.month
df['Season'] = df['Month'] % 12 // 3 + 1  

# 6. Modeling with seasonal features
X = df[['Temperature', 'DayOfYear', 'Season']]
y = df['Consumption']

model = LinearRegression()
model.fit(X, y)
df['Predicted'] = model.predict(X)

# 7. Residual analysis (normalized)
df['Residual'] = (df['Consumption'] - df['Predicted']) / 50000  

plt.figure(figsize=(12, 4))
df.plot(x='Date', y='Residual', title='Normalized Model Residuals (Range: -1 to +1)')
plt.axhline(0, color='r', linestyle='--', alpha=0.7)
plt.ylim(-1, 1)  
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8. Final evaluation
print(f"Model R²: {model.score(X, y):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y, df['Predicted'])):.0f} MW")

# 9. Compare with naive seasonal mean
seasonal_avg = df.groupby('DayOfYear')['Consumption'].mean()
df['Seasonal_Baseline'] = df['DayOfYear'].map(seasonal_avg)
print(f"\nNaive Seasonal Model RMSE: {np.sqrt(mean_squared_error(df['Consumption'], df['Seasonal_Baseline'])):.0f} MW")