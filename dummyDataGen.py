import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_water_quality_dummy_data(start_date, num_days=365):
    """
    Generate daily dummy water quality data
    
    Args:
    start_date (datetime): Starting date for data generation
    num_days (int): Number of days of data to generate
    
    Returns:
    pd.DataFrame: Generated water quality data
    """
    # Initialize empty lists to store data
    dates = []
    ph_values = []
    turbidity_values = []
    tds_values = []
    latitudes = []
    longitudes = []
    
    # Generate data for specified number of days
    current_date = start_date
    for _ in range(num_days):
        # Simulate some variability in water quality parameters with daily fluctuations
        
        # PH values typically range between 6.5 and 8.5
        # Add slight daily variation
        ph = round(max(6.5, min(8.5, np.random.normal(7.5, 0.5))), 2)
        
        # Turbidity in NTU (Nephelometric Turbidity Units)
        # Typical range: 0.5 - 10 NTU with daily fluctuations
        turbidity = round(np.random.uniform(0.5, 10), 2)
        
        # Total Dissolved Solids (TDS) in ppm
        # Typical range: 0 - 1000 ppm with daily variations
        tds = round(np.random.uniform(0, 1000), 2)
        
        # Latitude and Longitude (example location: somewhere in the continental US)
        # Keep location consistent for this dataset
        latitude = round(np.random.uniform(25, 50), 4)
        longitude = round(np.random.uniform(-125, -65), 4)
        
        # Store values
        dates.append(current_date)
        ph_values.append(ph)
        turbidity_values.append(turbidity)
        tds_values.append(tds)
        latitudes.append(latitude)
        longitudes.append(longitude)
        
        # Increment by one day
        current_date += timedelta(days=1)
    
    # Create DataFrame
    water_quality_data = pd.DataFrame({
        'date': dates,
        'ph': ph_values,
        'turbidity': turbidity_values,
        'tds': tds_values,
        'latitude': latitudes,
        'longitude': longitudes
    })
    
    return water_quality_data

# Example usage
def main():
    # Generate dummy data starting from a specific date
    start_date = datetime(2023, 1, 1)
    
    # Generate daily data for one year
    water_data = generate_water_quality_dummy_data(start_date, num_days=365)
    
    # Save to CSV (optional)
    water_data.to_csv('dummy_data.csv', index=False)
    
    # Print first few rows to verify
    print(water_data.head())
    
    # Print total number of records
    print(f"\nTotal number of daily records: {len(water_data)}")

if __name__ == "__main__":
    main()