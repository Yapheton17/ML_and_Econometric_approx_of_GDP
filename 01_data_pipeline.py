import pandas as pd
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

def fetch_and_clean_data(api_key):
    fred = Fred(api_key=api_key)
    
    # 1. Fetch data starting from 1990
    print("Fetching data from FRED...")
    # Real Gross Domestic Product (GDPC1) is standard for modeling, not nominal
    gdp = fred.get_series('GDPC1', observation_start='1990-01-01')
    unrate = fred.get_series('UNRATE', observation_start='1990-01-01')
    cpi = fred.get_series('CPIAUCSL', observation_start='1990-01-01')
    oil = fred.get_series('WTISPLC', observation_start='1990-01-01') # WTI Crude
    
    # 2. Convert to DataFrames
    df_gdp = pd.DataFrame(gdp, columns=['Real_GDP'])
    df_unrate = pd.DataFrame(unrate, columns=['Unemployment'])
    df_cpi = pd.DataFrame(cpi, columns=['CPI'])
    df_oil = pd.DataFrame(oil, columns=['Oil_Price'])
    
    # 3. Align frequencies to Quarterly (QS = Quarter Start)
    print("Aligning frequencies to Quarterly...")
    df_unrate_q = df_unrate.resample('QS').mean()
    df_cpi_q = df_cpi.resample('QS').mean()
    df_oil_q = df_oil.resample('QS').mean()
    df_gdp_q = df_gdp.resample('QS').mean() 
    
    # 4. Merge all into one Master DataFrame
    master_df = pd.concat([df_gdp_q, df_unrate_q, df_cpi_q, df_oil_q], axis=1)
    
    # 5. Feature Engineering (Raw numbers are bad for time series; we need rates of change)
    print("Engineering economic features...")
    master_df['GDP_Growth'] = master_df['Real_GDP'].pct_change() * 100
    master_df['Inflation_Rate'] = master_df['CPI'].pct_change(periods=4) * 100 # Year-over-Year
    master_df['Unemployment_Change'] = master_df['Unemployment'].diff()
    master_df['Oil_Growth'] = master_df['Oil_Price'].pct_change() * 100
    
    # 6. Create the Target Variable (Y)
    # We want to predict NEXT quarter's GDP Growth based on THIS quarter's features
    master_df['Target_Next_Quarter_GDP_Growth'] = master_df['GDP_Growth'].shift(-1)
    
    # Clean up missing values created by shifting and pct_change
    master_df.dropna(inplace=True)
    master_df.index.name = 'Date'
    
    return master_df

if __name__ == "__main__":
    # Insert your FRED key here
    API_KEY = '54fcfe2acde60a42116b51b9dca79680'
    
    final_data = fetch_and_clean_data(API_KEY)
    final_data.to_csv('master_data.csv')
    print("\nPipeline complete! Saved as 'master_data.csv'")
    print(final_data[['GDP_Growth', 'Target_Next_Quarter_GDP_Growth']].tail())