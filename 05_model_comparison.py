import pandas as pd
import numpy as np
import statsmodels.stats.diagnostic as diag

def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred)**2))

def run_final_comparison():
    # 1. Load the saved predictions
    try:
        df3 = pd.read_csv('03_predictions.csv', index_col='Date', parse_dates=True)
        df4 = pd.read_csv('04_predictions.csv', index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print("Error: Run scripts 03 and 04 first to generate the prediction CSVs.")
        return
        
    actual = df3['Actual']
    
    # Dictionary of all model predictions
    # Dictionary of all model predictions
    models = {
        'OLS (Econometric)': df3['OLS_Pred'],
        'ARIMAX (Econometric)': df3['ARIMAX_Pred'],
        'Random Forest (ML)': df4['RF_Pred'],
        'Gradient Boosting (ML)': df4['GB_Pred'] # Updated this line!
    }
    
    # 2. Evaluate Accuracy (RMSE)
    print("\n=== OUT-OF-SAMPLE ACCURACY (RMSE) ===")
    print("Lower is better.\n")
    for name, pred in models.items():
        error = rmse(actual, pred)
        print(f"{name.ljust(22)}: {error:.3f}")
        
    # 3. Ljung-Box Test on Residuals
    print("\n=== LJUNG-BOX TEST FOR WHITE NOISE ===")
    print("Null Hypothesis: The errors are random (model captured all patterns).")
    print("If p-value < 0.05, the model failed to capture all underlying patterns.\n")
    
    for name, pred in models.items():
        residuals = actual - pred
        # Test for autocorrelation up to 4 lags (1 year of quarters)
        lb_test = diag.acorr_ljungbox(residuals, lags=[4], return_df=True)
        p_val = lb_test['lb_pvalue'].iloc[0]
        
        status = "Failed (Patterns remain)" if p_val < 0.05 else "Passed (White Noise)"
        print(f"{name.ljust(22)} p-value: {p_val:.4f} -> {status}")

if __name__ == "__main__":
    run_final_comparison()