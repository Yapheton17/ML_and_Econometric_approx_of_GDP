import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred)**2))

def run_univariate_models():
    # 1. Load Data
    df = pd.read_csv('master_data.csv', index_col='Date', parse_dates=True)
    ts_data = df['GDP_Growth']
    
    # 2. Train/Test Split (Keep last 12 quarters / 3 years for testing)
    train = ts_data.iloc[:-12]
    test = ts_data.iloc[-12:]
    print(f"Training on {len(train)} quarters, testing on {len(test)} quarters.")

    # 3. Naive Forecast (Guessing next quarter is identical to last observed quarter)
    naive_pred = [train.iloc[-1]] * len(test)
    
    # 4. Exponential Smoothing (ETS)
    print("Fitting ETS...")
    ets_model = ExponentialSmoothing(train, trend='add', seasonal=None)
    ets_fit = ets_model.fit()
    ets_pred = ets_fit.forecast(len(test))
    
    # 5. ARIMA (order p=1, d=0, q=1 since GDP Growth is stationary)
    print("Fitting ARIMA...")
    arima_model = SARIMAX(train, order=(1, 0, 1))
    arima_fit = arima_model.fit(disp=False)
    arima_pred = arima_fit.forecast(len(test))

    # 6. Evaluation
    print(f"\n--- OOS RMSE SCORES ---")
    print(f"Naive: {rmse(test, naive_pred):.3f}")
    print(f"ETS:   {rmse(test, ets_pred):.3f}")
    print(f"ARIMA: {rmse(test, arima_pred):.3f}")
    
    # 7. Plotting the Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(train.index[-20:], train[-20:], label='Training Data (Last 20 Qs)', color='gray')
    plt.plot(test.index, test, label='Actual GDP Growth', color='black', marker='o', linewidth=2)
    
    plt.plot(test.index, naive_pred, label='Naive Forecast', linestyle=':')
    plt.plot(test.index, ets_pred, label='ETS', linestyle='--')
    plt.plot(test.index, arima_pred, label='ARIMA', linestyle='-.')
    
    plt.title('02 - Traditional Univariate Forecasts (Out-Of-Sample)')
    plt.ylabel('GDP Growth %')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('02_univariate_oos.png')
    print("\nSaved forecast plot to '02_univariate_oos.png'")

if __name__ == "__main__":
    run_univariate_models()