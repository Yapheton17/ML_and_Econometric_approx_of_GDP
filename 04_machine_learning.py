import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def run_ml_models():
    # 1. Load Data
    try:
        df = pd.read_csv('master_data.csv', index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print("Error: Could not find 'master_data.csv'. Run script 01 first.")
        return
        
    y = df['Target_Next_Quarter_GDP_Growth']
    # The ML models use all current features to predict the future target
    X = df.drop(columns=['Target_Next_Quarter_GDP_Growth'])
    
    # 2. Train/Test Split (Chronological split for time-series)
    X_train, X_test = X.iloc[:-12], X.iloc[-12:]
    y_train, y_test = y.iloc[:-12], y.iloc[-12:]
    
    # 3. Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # 4. Train Gradient Boosting (Our BART Replacement)
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    
    # 5. Save Predictions for Script 05
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'RF_Pred': rf_pred,
        'GB_Pred': gb_pred
    }, index=y_test.index)
    predictions_df.to_csv('04_predictions.csv')
    
    # 6. Plot Out-Of-Sample Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual GDP Growth', color='black', marker='o', linewidth=2)
    plt.plot(y_test.index, rf_pred, label='Random Forest', color='blue', linestyle='--')
    plt.plot(y_test.index, gb_pred, label='Gradient Boosting', color='green', linestyle='-.')
    
    plt.title('04 - Machine Learning Forecasts (Out-Of-Sample)')
    plt.xlabel('Date')
    plt.ylabel('GDP Growth %')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('04_ml_oos.png')
    print("\nSaved plot to '04_ml_oos.png' and predictions to '04_predictions.csv'")

if __name__ == "__main__":
    run_ml_models()