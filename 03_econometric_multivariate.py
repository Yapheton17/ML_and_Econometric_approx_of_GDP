import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def run_econometric_models():
    # 1. Load the clean data
    df = pd.read_csv('master_data.csv', index_col='Date', parse_dates=True)
    
    # 2. Define Target and Exogenous Features
    y = df['Target_Next_Quarter_GDP_Growth']
    # Exogenous variables: Inflation, Unemployment Change, and Oil Price Growth
    X = df[['Inflation_Rate', 'Unemployment_Change', 'Oil_Growth']]
    
    # 3. Train/Test Split (Chronological - last 12 quarters for testing)
    train_y, test_y = y.iloc[:-12], y.iloc[-12:]
    train_X, test_X = X.iloc[:-12], X.iloc[-12:]
    
    # 4. OLS Regression (Testing Economic Theory)
    print("Fitting OLS Regression...")
    train_X_sm = sm.add_constant(train_X) # OLS requires a constant
    test_X_sm = sm.add_constant(test_X)
    
    ols_model = sm.OLS(train_y, train_X_sm).fit()
    ols_pred = ols_model.predict(test_X_sm)
    
    # 5. ARIMAX (ARIMA + Exogenous Variables)
    print("Fitting ARIMAX...")
    arimax_model = SARIMAX(train_y, exog=train_X, order=(1, 0, 1)).fit(disp=False)
    arimax_pred = arimax_model.forecast(len(test_y), exog=test_X)
    
    # 6. Save Predictions for Final Comparison
    predictions_df = pd.DataFrame({
        'Actual': test_y,
        'OLS_Pred': ols_pred,
        'ARIMAX_Pred': arimax_pred
    }, index=test_y.index)
    predictions_df.to_csv('03_predictions.csv')
    
    # 7. Plot Out-Of-Sample Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(train_y.index[-20:], train_y[-20:], label='Training Data', color='gray')
    plt.plot(test_y.index, test_y, label='Actual GDP Growth', color='black', marker='o', linewidth=2)
    plt.plot(test_y.index, ols_pred, label='OLS (Exogenous)', linestyle='--')
    plt.plot(test_y.index, arimax_pred, label='ARIMAX', linestyle='-.')
    
    plt.title('03 - Econometric Forecasts (Out-Of-Sample)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('03_econometric_oos.png')
    print("\nSaved plot to '03_econometric_oos.png' and predictions to '03_predictions.csv'")

if __name__ == "__main__":
    run_econometric_models()