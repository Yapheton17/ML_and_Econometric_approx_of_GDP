import pandas as pd
from fredapi import Fred

# Initialize the FRED client with your API key
API_KEY = '54fcfe2acde60a42116b51b9dca79680'
fred = Fred(api_key=API_KEY)

def fetch_economic_series(fred_client, series_id, start_date='2000-01-01'):
    """
    Fetches historical data for a specific FRED series and returns a pandas DataFrame.
    """
    print(f"Fetching data for {series_id}...")
    
    # Fetch the series
    data = fred_client.get_series(series_id, observation_start=start_date)
    
    # Convert the pandas Series to a DataFrame
    df = pd.DataFrame(data, columns=[series_id])
    df.index.name = 'Date'
    
    return df

def build_gdp_dataset(fred_client):
    """
    Builds a dataset containing GDP and a few potential predictor variables.
    """
    # Series IDs: GDP, Unemployment Rate (UNRATE), and Consumer Price Index (CPIAUCSL)
    series_to_pull = ['GDP', 'UNRATE', 'CPIAUCSL']
    
    df_list = []
    for series in series_to_pull:
        df = fetch_economic_series(fred_client, series)
        df_list.append(df)
        
    # Merge all the dataframes on the Date index
    # Using 'outer' join because GDP is quarterly, while UNRATE and CPI are monthly
    merged_df = pd.concat(df_list, axis=1)
    
    # Forward-fill the quarterly GDP data so it aligns with monthly data
    merged_df.ffill(inplace=True)
    
    # Drop rows with any missing values at the beginning of the dataset
    merged_df.dropna(inplace=True)
    
    return merged_df

# Generate the dataset
gdp_data = build_gdp_dataset(fred)
print(gdp_data.tail())