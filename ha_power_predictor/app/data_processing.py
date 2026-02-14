"""
Data processing functions for Home Assistant power data.
Handles parsing from HA API, binning, and feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from typing import List, Dict, Any


def process_ha_data(
    power_data: List[Dict[str, Any]],
    temp_data: List[Dict[str, Any]],
    bin_size_minutes: int = 60,
    timezone: str = 'UTC'
) -> pd.DataFrame:
    """
    Process Home Assistant API data into binned DataFrame.
    
    Args:
        power_data: List of power state dicts from HA API
        temp_data: List of temperature state dicts from HA API
        bin_size_minutes: Size of time bins in minutes
        timezone: Timezone for timestamp localization
    
    Returns:
        DataFrame with binned power and temperature data plus temporal features
    """
    print(f"\n📊 Processing HA data with {bin_size_minutes}-minute bins...")
    
    # Convert power data to DataFrame
    power_records = []
    for record in power_data:
        try:
            timestamp = pd.to_datetime(record.get('last_changed') or record.get('last_updated'))
            state = record.get('state')
            
            if state not in ['unknown', 'unavailable', None]:
                power_records.append({
                    'timestamp': timestamp,
                    'consumption': float(state)
                })
        except (ValueError, TypeError):
            continue
    
    if not power_records:
        raise ValueError("No valid power records found")
    
    power_df = pd.DataFrame(power_records)
    
    # Convert temperature data to DataFrame
    temp_records = []
    for record in temp_data:
        try:
            timestamp = pd.to_datetime(record.get('last_changed') or record.get('last_updated'))
            state = record.get('state')
            
            if state not in ['unknown', 'unavailable', None]:
                temp_records.append({
                    'timestamp': timestamp,
                    'temperature': float(state)
                })
        except (ValueError, TypeError):
            continue
    
    if not temp_records:
        raise ValueError("No valid temperature records found")
    
    temp_df = pd.DataFrame(temp_records)
    
    print(f"  - Found {len(power_df)} power readings")
    print(f"  - Found {len(temp_df)} temperature readings")
    
    # Localize timestamps
    tz = pytz.timezone(timezone)
    power_df['timestamp'] = power_df['timestamp'].dt.tz_convert(tz)
    temp_df['timestamp'] = temp_df['timestamp'].dt.tz_convert(tz)
    
    # Create time bins
    bin_freq = f'{bin_size_minutes}min'
    
    power_df['time_bin'] = power_df['timestamp'].dt.floor(bin_freq)
    power_binned = power_df.groupby('time_bin')['consumption'].mean().reset_index()
    
    temp_df['time_bin'] = temp_df['timestamp'].dt.floor(bin_freq)
    temp_binned = temp_df.groupby('time_bin')['temperature'].mean().reset_index()
    
    # Merge power and temperature data
    df_merged = pd.merge(power_binned, temp_binned, on='time_bin', how='inner')
    df_merged = df_merged.rename(columns={'time_bin': 'timestamp'})
    
    # Sort by timestamp
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
    
    # Add temporal features
    df_merged['year'] = df_merged['timestamp'].dt.year
    df_merged['month'] = df_merged['timestamp'].dt.month
    df_merged['day_of_week'] = df_merged['timestamp'].dt.dayofweek
    df_merged['hour'] = df_merged['timestamp'].dt.hour
    df_merged['minute'] = df_merged['timestamp'].dt.minute
    
    print(f"  ✓ Created {len(df_merged)} time bins")
    print(f"  ✓ Date range: {df_merged['timestamp'].min()} to {df_merged['timestamp'].max()}")
    
    return df_merged


def add_lagged_features(
    df: pd.DataFrame,
    n_power_lags: int = 0,
    n_temp_lags: int = 0
) -> pd.DataFrame:
    """
    Add lagged features for previous power and temperature readings.
    
    Args:
        df: DataFrame with consumption and temperature columns
        n_power_lags: Number of previous power readings to include
        n_temp_lags: Number of previous temperature readings to include
    
    Returns:
        DataFrame with lagged features added
    """
    if n_power_lags == 0 and n_temp_lags == 0:
        return df
    
    print(f"  - Adding {n_power_lags} power lags, {n_temp_lags} temp lags...")
    
    df_lagged = df.copy()
    
    for i in range(1, n_power_lags + 1):
        df_lagged[f'power_lag_{i}'] = df_lagged['consumption'].shift(i)
    
    for i in range(1, n_temp_lags + 1):
        df_lagged[f'temp_lag_{i}'] = df_lagged['temperature'].shift(i)
    
    # Remove rows with NaN values from lagging
    max_lag = max(n_power_lags, n_temp_lags)
    if max_lag > 0:
        df_lagged = df_lagged.iloc[max_lag:].reset_index(drop=True)
    
    return df_lagged


def get_default_features() -> List[str]:
    """Get the default feature set for training."""
    return ['year', 'month', 'day_of_week', 'hour', 'temperature']
