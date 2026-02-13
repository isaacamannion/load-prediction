"""
Data processing functions for Home Assistant energy data.
Handles parsing, binning, and feature engineering.
"""

import pandas as pd
import numpy as np


def parse_ha_export(csv_file, bin_size_minutes=60):
    """
    Parse Home Assistant CSV export with power and temperature data.
    
    Args:
        csv_file: Path to CSV file exported from Home Assistant
        bin_size_minutes: Size of time bins in minutes (default: 60)
    
    Returns:
        DataFrame with binned power and temperature data plus temporal features
    """
    print(f"\n📊 Parsing data with {bin_size_minutes}-minute bins...")
    
    df = pd.read_csv(csv_file)
    
    df['timestamp_local'] = pd.to_datetime(df['last_changed'], utc=True).dt.tz_convert('Australia/Sydney')
    
    power_df = df[df['entity_id'] == 'sensor.sigen_plant_consumed_power'].copy()
    temp_df = df[df['entity_id'] == 'sensor.weather_temperature'].copy()
    
    print(f"  - Found {len(power_df)} power readings")
    print(f"  - Found {len(temp_df)} temperature readings")
    
    power_df['consumption'] = pd.to_numeric(power_df['state'], errors='coerce')
    power_df = power_df.dropna(subset=['consumption'])
    
    temp_df['temperature'] = pd.to_numeric(temp_df['state'], errors='coerce')
    temp_df = temp_df.dropna(subset=['temperature'])
    
    bin_freq = f'{bin_size_minutes}min'
    
    power_df['time_bin'] = power_df['timestamp_local'].dt.floor(bin_freq)
    power_binned = power_df.groupby('time_bin')['consumption'].mean().reset_index()
    
    temp_df['time_bin'] = temp_df['timestamp_local'].dt.floor(bin_freq)
    temp_binned = temp_df.groupby('time_bin')['temperature'].mean().reset_index()
    
    df_merged = pd.merge(power_binned, temp_binned, on='time_bin', how='inner')
    df_merged = df_merged.rename(columns={'time_bin': 'timestamp'})
    
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
    
    df_merged['year'] = df_merged['timestamp'].dt.year
    df_merged['month'] = df_merged['timestamp'].dt.month
    df_merged['day_of_week'] = df_merged['timestamp'].dt.dayofweek
    df_merged['hour'] = df_merged['timestamp'].dt.hour
    df_merged['minute'] = df_merged['timestamp'].dt.minute
    
    print(f"  ✓ Created {len(df_merged)} time bins")
    print(f"  ✓ Date range: {df_merged['timestamp'].min()} to {df_merged['timestamp'].max()}")
    
    return df_merged


def add_lagged_features(df, n_power_lags=0, n_temp_lags=0):
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
    
    max_lag = max(n_power_lags, n_temp_lags)
    if max_lag > 0:
        df_lagged = df_lagged.iloc[max_lag:].reset_index(drop=True)
    
    return df_lagged


def apply_smoothing(predictions, window_size, method='moving_average'):
    """
    Apply smoothing to predictions to reduce spikes.
    
    Args:
        predictions: Array of predictions
        window_size: Size of smoothing window
        method: 'moving_average' or 'exponential'
    
    Returns:
        Smoothed predictions
    """
    if method == 'moving_average':
        smoothed = pd.Series(predictions).rolling(window=window_size, center=True).mean()
        smoothed = smoothed.fillna(method='bfill').fillna(method='ffill')
    else:
        smoothed = pd.Series(predictions).ewm(span=window_size).mean()
    
    return smoothed.values


def get_default_features():
    """Get the default feature set for training."""
    return ['year', 'month', 'day_of_week', 'hour', 'temperature']
