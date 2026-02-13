# Energy Load Prediction System

A machine learning system that predicts household energy consumption using Home Assistant historical data. Uses Quantile Regression for conservative forecasting and optional Neural Networks for comparison.

## Features

- 🎯 **Quantile Regression**: Conservative predictions that avoid underestimation
- 🧠 **Neural Networks**: Optional comparison with standard regression
- 🕐 **Dynamic Quantiles**: Different prediction levels for peak vs off-peak hours
- 📊 **Iterative Forecasting**: Realistic simulation using predicted values for lag features
- 🔧 **Configurable**: Adjust time bins, lag features, quantiles, and more
- 📈 **Visualizations**: Automatic generation of overview and 48-hour forecast plots

## Quick Start

### 1. Export Your Data from Home Assistant

#### Step 1: Access Home Assistant History

1. Log into your Home Assistant instance
2. Navigate to history

#### Step 2: Export Power Consumption Data
#### Step 3: Export Temperature Data

1. Find your power consumption sensor (e.g., `sensor.sigen_plant_consumed_power`)
2. Click on the sensor to view its history
5. Choose a date range (recommended: at least 30 days for good predictions)
1. Find your temperature sensor (e.g., `sensor.weather_temperature`)
3. Click the **Download** button (top right of the graph)
4. Select **Download CSV**
3. Save the file

```

**Expected CSV format:**
```csv
entity_id,state,last_changed
sensor.sigen_plant_consumed_power,1.02,2026-01-12T22:00:00.000Z
sensor.sigen_plant_consumed_power,3.83,2026-01-12T23:00:00.000Z
sensor.weather_temperature,25.3,2026-01-12T22:00:00.000Z
sensor.weather_temperature,24.8,2026-01-12T23:00:00.000Z
```

**Important Notes:**
- Update the sensor entity IDs in `data_processing.py` (lines 40-41) to match your sensors
- The code expects timestamps in UTC
- Power values should be in kW (the code will handle conversion if needed)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.3.0

### 3. Run

#### Using Recommended Settings (Starting Point)

This uses:
- 60-minute time bins
- 10 power lag features
- 5 temperature lag features
- Dynamic quantile (75% for peak hours 9am-10pm, 50% for off-peak)
- No smoothing or buffer

## Command Line Arguments

```
--csv PATH              Path to Home Assistant CSV export (required)
--model TYPE            Model type: 'quantile', 'neural', or 'both' (default: quantile)
--bins MINUTES          Time bin size in minutes (default: 60)
--power-lags N          Number of previous power readings to use (default: 10)
--temp-lags N           Number of previous temperature readings to use (default: 5)
--quantile Q            Target quantile for predictions, 0.5-0.99 (default: 0.75)
--dynamic               Use different quantiles for peak/off-peak hours
--peak-start HOUR       Peak period start hour, 0-23 (default: 9)
--peak-end HOUR         Peak period end hour, 0-23 (default: 22)
--buffer PERCENT        Safety buffer percentage, 0-50 (default: 0)
--smooth                Apply moving average smoothing to predictions
--recommended           Use recommended configuration
```

## Understanding the Results

### Output Files

The system generates two PNG files:

1. **`energy_prediction_overview.png`**: Full test period showing all predictions
2. **`energy_prediction_48h.png`**: Detailed view of first 48 hours

### Metrics Explained

- **R² Score**: How well predictions match actual values (higher is better, 1.0 is perfect)
- **MAE (Mean Absolute Error)**: Average prediction error in kW (lower is better)
- **RMSE (Root Mean Squared Error)**: Prediction error with emphasis on large misses (lower is better)
- **Coverage**: Percentage of actual values below predictions (target: 75% for 75th percentile)

### Recommended Settings Explained

Based on experimentation, these settings provide good results:

```python
Bins: 60 minutes
  Why: Balances granularity with data stability

Power Lags: 10
  Why: Captures recent consumption patterns without overfitting

Temperature Lags: 3
  Why: Temperature changes more slowly than power consumption

Peak Hours: 9am-10pm
  Why: Captures typical high-usage period (morning prep + evening activities)

Peak Quantile: 75th percentile
  Why: Conservative without excessive over-prediction

Off-peak Quantile: 50th percentile
  Why: More accurate during low-usage periods

No Smoothing
  Why: Preserves spike detection (important for capacity planning)
```

## Customizing for Your Data

### Updating Sensor Names

Edit `data_processing.py` and update lines 40-41:

```python
# Change these to match your Home Assistant sensors
power_df = df[df['entity_id'] == 'sensor.YOUR_POWER_SENSOR'].copy()
temp_df = df[df['entity_id'] == 'sensor.YOUR_TEMP_SENSOR'].copy()
```

### Timezone Configuration

The code is set to `Australia/Sydney`. Update line 37 in `data_processing.py`:

```python
df['timestamp_local'] = pd.to_datetime(df['last_changed'], utc=True).dt.tz_convert('YOUR_TIMEZONE')
```

## Project Structure

```
load-prediction/
├── main.py                 # Main entry point with CLI
├── models.py              # Model classes (Quantile Regression, Neural Network)
├── data_processing.py     # Data loading and feature engineering
├── config.py              # Configuration classes and defaults
├── visualization.py       # Plotting and results display
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## How It Works

### 1. Data Processing
- Loads Home Assistant CSV export
- Bins data into time intervals (default: 60 minutes)
- Merges power consumption with temperature readings
- Adds  features (hour, day of week, month, year)
- Creates lag features from previous readings

### 2. Model Training

**Quantile Regression:**
- Trains to predict specified percentile (e.g., 75th)
- Produces conservative forecasts that avoid underestimation
- Can use different quantiles for peak vs off-peak hours

**Neural Network (optional):**
- Standard regression for comparison
- Multi-layer perceptron with early stopping
- Typically less conservative than quantile regression

### 3. Iterative Prediction
- Simulates real-time forecasting
- Uses predicted values (not actual) for lag features
- More realistic than using actual future values

### 4. Evaluation
- Calculates R², MAE, RMSE
- Measures coverage (% of actuals below predictions)
- Generates visualization plots


```

### Experimenting with Parameters

Try different combinations to find what works best for your data:

**For more conservative predictions:**
- Increase `--quantile` (e.g., 0.85 or 0.90)
- Add a safety buffer with `--buffer 10`

**For more accurate predictions:**
- Use `--quantile 0.50` (median)
- Increase lag features for pattern recognition

**For smoother predictions:**
- Add `--smooth` flag
- Reduce time bin size (e.g., `--bins 30`)

## Troubleshooting

### "No data found for sensor"
- Check sensor names in `data_processing.py` match your Home Assistant
- Verify CSV contains both power and temperature data

### "Not enough data after binning"
- Export more historical data (30+ days recommended)
- Try larger time bins (e.g., `--bins 120`)

### "Coverage is too low"
- Increase quantile (e.g., `--quantile 0.85`)
- Add safety buffer (e.g., `--buffer 10`)
- Check if your data has unusual spikes or patterns

### "R² score is negative"
- Model is performing worse than a simple mean
- Try different lag features
- Check data quality and date range

## Contributing

This is a research/experimental project. Feel free to:
- Experiment with different models or features
- Add support for additional sensors
- Improve the visualization
- Test with different home automation systems
