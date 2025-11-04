import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import numpy as np
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# List of European countries with their ISO codes and coordinates
EUROPEAN_COUNTRIES = {
    'Austria': {'code': 'AT', 'lat': 47.516, 'lon': 14.550},
    'Belgium': {'code': 'BE', 'lat': 50.503, 'lon': 4.469},
    'Bulgaria': {'code': 'BG', 'lat': 42.733, 'lon': 25.485},
    'Croatia': {'code': 'HR', 'lat': 45.100, 'lon': 15.200},
    'Czech Republic': {'code': 'CZ', 'lat': 49.817, 'lon': 15.472},
    'Denmark': {'code': 'DK', 'lat': 56.263, 'lon': 9.501},
    'Estonia': {'code': 'EE', 'lat': 58.595, 'lon': 25.013},
    'Finland': {'code': 'FI', 'lat': 61.924, 'lon': 25.748},
    'France': {'code': 'FR', 'lat': 46.227, 'lon': 2.213},
    'Germany': {'code': 'DE', 'lat': 51.165, 'lon': 10.451},
    'Greece': {'code': 'GR', 'lat': 39.074, 'lon': 21.824},
    'Hungary': {'code': 'HU', 'lat': 47.162, 'lon': 19.503},
    'Ireland': {'code': 'IE', 'lat': 53.412, 'lon': -8.243},
    'Italy': {'code': 'IT', 'lat': 41.871, 'lon': 12.567},
    'Latvia': {'code': 'LV', 'lat': 56.879, 'lon': 24.603},
    'Lithuania': {'code': 'LT', 'lat': 55.169, 'lon': 23.881},
    'Luxembourg': {'code': 'LU', 'lat': 49.815, 'lon': 6.129},
    'Netherlands': {'code': 'NL', 'lat': 52.132, 'lon': 5.291},
    'Poland': {'code': 'PL', 'lat': 51.919, 'lon': 19.145},
    'Portugal': {'code': 'PT', 'lat': 39.399, 'lon': -8.224},
    'Romania': {'code': 'RO', 'lat': 45.943, 'lon': 24.966},
    'Slovakia': {'code': 'SK', 'lat': 48.669, 'lon': 19.699},
    'Slovenia': {'code': 'SI', 'lat': 46.151, 'lon': 14.995},
    'Spain': {'code': 'ES', 'lat': 40.463, 'lon': -3.749},
    'Sweden': {'code': 'SE', 'lat': 60.128, 'lon': 18.643}
}

def create_session_with_retries():
    """Create a session with retry strategy"""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def get_weather_data(session, lat, lon, start_date, end_date):
    """Fetch historical weather data from Open-Meteo API with rate limiting"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': ['temperature_2m_mean', 'windspeed_10m_mean'],
        'timezone': 'UTC'
    }
    
    try:
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            'temperature': data['daily']['temperature_2m_mean'],
            'wind_speed': data['daily']['windspeed_10m_mean']
        })
        return df
    
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def generate_gas_demand(weather_df, country_code):
    """Generate synthetic gas demand data based on weather patterns"""
    # Parameters for gas demand simulation
    base_demand = {
        'small': {'mean': 5000, 'std': 1000},    # Small countries
        'medium': {'mean': 15000, 'std': 3000},  # Medium countries
        'large': {'mean': 30000, 'std': 6000}    # Large countries
    }
    
    # Classify countries by size
    large_countries = ['DE', 'FR', 'IT', 'GB', 'ES', 'PL']
    medium_countries = ['NL', 'BE', 'SE', 'AT', 'CZ', 'RO', 'PT', 'GR', 'HU']
    
    # Select base demand parameters
    if country_code in large_countries:
        params = base_demand['large']
    elif country_code in medium_countries:
        params = base_demand['medium']
    else:
        params = base_demand['small']
    
    # Generate daily demand with seasonal and temperature effects
    df = weather_df.copy()
    
    # Add seasonal component (higher in winter)
    df['day_of_year'] = df['date'].dt.dayofyear
    seasonal_factor = np.sin(2 * np.pi * (df['day_of_year'] - 345) / 365)
    
    # Temperature effect (more gas used when colder)
    temp_factor = (15 - df['temperature']) / 20
    
    # Base demand with random variation
    base = np.random.normal(params['mean'], params['std'], len(df))
    
    # Combine factors
    df['gas_demand'] = base * (1 + 0.5 * seasonal_factor + 0.3 * temp_factor)
    
    # Add yearly growth trend (2% per year)
    years_since_start = (df['date'].dt.year - 2013)
    df['gas_demand'] *= (1.02 ** years_since_start)
    
    # Add some random noise
    df['gas_demand'] += np.random.normal(0, params['std'] * 0.1, len(df))
    
    # Ensure no negative values
    df['gas_demand'] = df['gas_demand'].clip(lower=0)
    
    return df[['date', 'gas_demand']]

def main():
    start_date = "2013-01-01"
    end_date = "2024-01-01"
    
    session = create_session_with_retries()
    all_data = []
    
    for country, info in tqdm(EUROPEAN_COUNTRIES.items()):
        print(f"\nProcessing {country}...")
        
        # Get weather data
        weather_df = get_weather_data(
            session,
            info['lat'], 
            info['lon'], 
            start_date, 
            end_date
        )
        
        if weather_df is None:
            print(f"Skipping {country} due to weather data error")
            continue
        
        # Generate synthetic gas demand data
        gas_df = generate_gas_demand(weather_df, info['code'])
        
        # Combine data
        combined_df = pd.merge(
            weather_df, 
            gas_df, 
            on='date', 
            how='outer'
        )
        
        combined_df['country'] = country
        all_data.append(combined_df)
        
        # Add delay between countries to avoid weather API rate limits
        time.sleep(2)
    
    if all_data:
        # Combine all country data
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Clean and sort the data
        final_df = final_df.sort_values(['country', 'date'])
        final_df = final_df.fillna(method='ffill')
        
        # Save to CSV
        output_file = 'european_gas_demand_weather_data.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        
        # Print summary statistics
        print("\nDataset Summary:")
        print(f"Total records: {len(final_df)}")
        print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        print(f"Countries: {final_df['country'].nunique()}")
        print("\nMean gas demand by country:")
        print(final_df.groupby('country')['gas_demand'].mean().sort_values(ascending=False))
    else:
        print("\nNo data was collected. Please check the error messages above.")

if __name__ == "__main__":
    main() 