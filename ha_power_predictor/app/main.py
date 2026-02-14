"""
Main Flask application for HA Power Predictor add-on.
Provides web UI and API endpoints for power prediction.
"""

from flask import Flask, render_template, jsonify, request
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import traceback

from .ha_client import HomeAssistantClient
from .data_processing import process_ha_data, add_lagged_features, get_default_features
from .models import QuantileRegressionModel, predict_iterative
from .config import get_config_from_env

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Global state
prediction_results = None
last_prediction_time = None


@app.route('/')
def index():
    """Render main page."""
    config = get_config_from_env()
    return render_template('index.html', config=config)


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    config = get_config_from_env()
    return jsonify(config)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Run power prediction model.
    
    Returns predictions and model performance metrics.
    """
    global prediction_results, last_prediction_time
    
    try:
        # Get configuration
        config = get_config_from_env()
        
        # Initialize HA client
        ha_client = HomeAssistantClient(
            base_url=os.getenv('HA_URL', 'http://supervisor/core'),
            token=os.getenv('SUPERVISOR_TOKEN')
        )
        
        # Fetch historical data
        app.logger.info(f"Fetching {config['history_days']} days of history...")
        end_time = datetime.now()
        start_time = end_time - timedelta(days=config['history_days'])
        
        power_data = ha_client.get_history(
            config['power_entity'],
            start_time,
            end_time
        )
        
        temp_data = ha_client.get_history(
            config['temperature_entity'],
            start_time,
            end_time
        )
        
        if not power_data or not temp_data:
            return jsonify({'error': 'No historical data found'}), 400
        
        # Process data
        app.logger.info("Processing data...")
        df = process_ha_data(
            power_data,
            temp_data,
            bin_size_minutes=config['bin_size_minutes'],
            timezone=config['timezone']
        )
        
        if len(df) < 100:
            return jsonify({'error': f'Insufficient data: only {len(df)} records'}), 400
        
        # Add lagged features
        df = add_lagged_features(
            df,
            n_power_lags=config['n_power_lags'],
            n_temp_lags=config['n_temp_lags']
        )
        
        # Split train/test
        train_size = int(len(df) * config['train_percentage'] / 100)
        df_train = df.iloc[:train_size].copy()
        df_test = df.iloc[train_size:].copy()
        
        # Prepare features
        features = get_default_features()
        
        # Add lag features to feature list
        for i in range(1, config['n_power_lags'] + 1):
            features.append(f'power_lag_{i}')
        for i in range(1, config['n_temp_lags'] + 1):
            features.append(f'temp_lag_{i}')
        
        X_train = df_train[features].values
        y_train = df_train['consumption'].values
        X_test = df_test[features].values
        y_test = df_test['consumption'].values
        
        hours_train = df_train['hour'].values
        hours_test = df_test['hour'].values
        
        # Train model
        app.logger.info("Training model...")
        dynamic_config = None
        if config['use_dynamic_quantile']:
            dynamic_config = {
                'peak_start': config['peak_start'],
                'peak_end': config['peak_end'],
                'peak_quantile': config['peak_quantile'],
                'offpeak_quantile': config['offpeak_quantile']
            }
        
        model = QuantileRegressionModel(
            quantile=config['quantile'],
            dynamic_config=dynamic_config
        )
        
        model.train(X_train, y_train, hours_train)
        
        # Make predictions
        app.logger.info("Making predictions...")
        result = predict_iterative(
            X_test, y_test, model, features,
            config['n_power_lags'], hours_test
        )
        
        y_pred = result['predictions']
        
        # Calculate metrics
        metrics = model.evaluate(y_test, y_pred)
        coverage = model.calculate_coverage(y_test, y_pred)
        
        # Prepare results
        predictions_list = []
        for i, row in df_test.iterrows():
            idx = i - train_size
            predictions_list.append({
                'timestamp': row['timestamp'].isoformat(),
                'actual': float(y_test[idx]),
                'predicted': float(y_pred[idx])
            })
        
        prediction_results = {
            'predictions': predictions_list,
            'metrics': {
                'r2': float(metrics['r2']),
                'mae': float(metrics['mae']),
                'rmse': float(metrics['rmse']),
                'coverage': float(coverage)
            },
            'model_info': {
                'type': 'Quantile Regression',
                'quantile': config['quantile'],
                'use_dynamic': config['use_dynamic_quantile'],
                'peak_hours': f"{config['peak_start']}-{config['peak_end']}" if config['use_dynamic_quantile'] else None,
                'peak_quantile': config['peak_quantile'] if config['use_dynamic_quantile'] else None,
                'offpeak_quantile': config['offpeak_quantile'] if config['use_dynamic_quantile'] else None
            },
            'data_info': {
                'train_samples': int(train_size),
                'test_samples': int(len(df_test)),
                'features': features,
                'history_days': config['history_days']
            }
        }
        
        last_prediction_time = datetime.now()
        
        # Publish predictions to HA
        try:
            ha_client.create_prediction_sensors(
                predictions_list[:48],  # Next 48 hours
                config['power_entity']
            )
            app.logger.info("Published predictions to Home Assistant")
        except Exception as e:
            app.logger.error(f"Failed to publish to HA: {e}")
        
        return jsonify(prediction_results)
        
    except Exception as e:
        app.logger.error(f"Prediction error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Get add-on status."""
    return jsonify({
        'status': 'running',
        'last_prediction': last_prediction_time.isoformat() if last_prediction_time else None,
        'has_results': prediction_results is not None
    })


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get last prediction results."""
    if prediction_results is None:
        return jsonify({'error': 'No predictions available'}), 404
    
    return jsonify(prediction_results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8099, debug=True)
