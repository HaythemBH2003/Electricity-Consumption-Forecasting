# Electricity-Consumption-Forecasting
This is an academic project that analyzes and forecasts electricity consumption using time series decomposition, autocorrelation analysis, and machine learning models. The code generates synthetic data with realistic seasonal patterns, performs diagnostic checks, and builds predictive models.
Key Features
------------
- Synthetic data generation with customizable seasonal patterns
- Time series decomposition (STL and classical methods)
- Autocorrelation (ACF) analysis
- Multiple forecasting models (Linear Regression, Ridge, Random Forest)
- Comprehensive visualization and diagnostic tools
- Residual analysis and model evaluation

Requirements
------------
- Python 3.8+
- Required packages:
  numpy
  pandas
  matplotlib
  seaborn
  scikit-learn
  statsmodels
  holidays

Usage
-----
1. Data Generation:
   df = generate_electricity_data(n_days=1095)  # 3 years of data

2. Feature Engineering:
   df = create_features(df)  # Adds time-based and lag features

3. Time Series Analysis:
   plot_diagnostics(df)  # ACF/PACF plots
   decompose_timeseries(df, period=365)  # Seasonal decomposition

4. Model Training:
   results, test = train_evaluate_models(df)

5. Visualization:
   plot_results(test, results)  # Forecast vs actual
   plot_feature_importance(results)  # For tree-based models

Key Functions
-------------
- generate_electricity_data(): Creates synthetic consumption data
- create_features(): Engineers temporal features
- plot_diagnostics(): Generates ACF plot
- train_evaluate_models(): Trains and compares ML models
- decompose_timeseries(): Flexible seasonal decomposition

Output Examples
---------------
1. Decomposition Plot:
   - Shows observed data, trend, seasonality, and residuals

2. ACF Diagnostics:
   - Identifies significant lags and seasonality periods

3. Forecast Visualization:
   - Compares actual vs predicted values

Customization
-------------
- Adjust generate_electricity_data() parameters to modify:
  - Seasonal amplitude
  - Noise level
  - Base consumption level

- Modify create_features() to:
  - Add/exclude specific lag features
  - Incorporate external variables

Model Interpretation
-------------------
- Linear Models: Check coefficients for feature importance
- Tree-based Models: Use plot_feature_importance()
- Residual Analysis: Verify randomness in residuals

