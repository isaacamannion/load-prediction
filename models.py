"""
Model implementations for energy load prediction.
Supports both Quantile Regression and Neural Network models.
"""

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class EnergyPredictionModel:
    """Base class for energy prediction models."""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler = None
        
    def train(self, X_train, y_train):
        """Train the model."""
        raise NotImplementedError
        
    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError
        
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        return {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def calculate_coverage(self, y_true, y_pred):
        """Calculate percentage of actual values below predictions."""
        return np.mean(y_true <= y_pred) * 100


class QuantileRegressionModel(EnergyPredictionModel):
    """Quantile Regression model for conservative forecasting."""
    
    def __init__(self, quantile=0.75, dynamic_config=None):
        super().__init__("Quantile Regression")
        self.quantile = quantile
        self.dynamic_config = dynamic_config
        self.models_by_hour = {}
        
    def train(self, X_train, y_train, hours_train=None):
        """
        Train the quantile regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            hours_train: Hour values for dynamic quantile training
        """
        if self.dynamic_config is not None and hours_train is not None:
            print(f"  - Training Dynamic Quantile Regression...")
            self._train_dynamic(X_train, y_train, hours_train)
        else:
            print(f"  - Training Quantile Regression (q={self.quantile:.2f})...")
            self.model = QuantileRegressor(
                quantile=self.quantile,
                alpha=0.01,
                solver='highs'
            )
            self.model.fit(X_train, y_train)
            
    def _train_dynamic(self, X_train, y_train, hours_train):
        """Train separate models for peak and off-peak hours."""
        peak_start = self.dynamic_config['peak_start']
        peak_end = self.dynamic_config['peak_end']
        peak_quantile = self.dynamic_config['peak_quantile']
        offpeak_quantile = self.dynamic_config['offpeak_quantile']
        
        peak_mask = (hours_train >= peak_start) & (hours_train <= peak_end)
        offpeak_mask = ~peak_mask
        
        if np.any(peak_mask):
            print(f"    - Peak hours ({peak_start}-{peak_end}): q={peak_quantile:.2f}, {np.sum(peak_mask)} samples")
            self.models_by_hour['peak'] = QuantileRegressor(
                quantile=peak_quantile, alpha=0.01, solver='highs'
            )
            self.models_by_hour['peak'].fit(X_train[peak_mask], y_train[peak_mask])
            
        if np.any(offpeak_mask):
            print(f"    - Off-peak hours: q={offpeak_quantile:.2f}, {np.sum(offpeak_mask)} samples")
            self.models_by_hour['offpeak'] = QuantileRegressor(
                quantile=offpeak_quantile, alpha=0.01, solver='highs'
            )
            self.models_by_hour['offpeak'].fit(X_train[offpeak_mask], y_train[offpeak_mask])
    
    def predict(self, X, hours=None):
        """Make predictions using appropriate model(s)."""
        if self.dynamic_config is not None and hours is not None:
            return self._predict_dynamic(X, hours)
        else:
            return self.model.predict(X)
    
    def _predict_dynamic(self, X, hours):
        """Predict using dynamic quantile models."""
        peak_start = self.dynamic_config['peak_start']
        peak_end = self.dynamic_config['peak_end']
        
        predictions = np.zeros(len(X))
        peak_mask = (hours >= peak_start) & (hours <= peak_end)
        offpeak_mask = ~peak_mask
        
        if np.any(peak_mask) and 'peak' in self.models_by_hour:
            predictions[peak_mask] = self.models_by_hour['peak'].predict(X[peak_mask])
            
        if np.any(offpeak_mask) and 'offpeak' in self.models_by_hour:
            predictions[offpeak_mask] = self.models_by_hour['offpeak'].predict(X[offpeak_mask])
            
        return predictions


class NeuralNetworkModel(EnergyPredictionModel):
    """Neural Network model for energy prediction."""
    
    def __init__(self, hidden_layers=(100, 50), max_iter=500):
        super().__init__("Neural Network")
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, hours_train=None):
        """Train the neural network model."""
        print(f"  - Training Neural Network (layers={self.hidden_layers})...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.model.fit(X_train_scaled, y_train)
        
    def predict(self, X, hours=None):
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def predict_iterative(X_test, y_test, model, features, n_power_lags, hours_test=None):
    """
    Perform iterative prediction, using predicted values for power lags.
    Simulates real-time forecasting where future actual power values aren't known.
    
    Args:
        X_test: Test features
        y_test: Test targets (for evaluation)
        model: Trained model (QuantileRegressionModel or NeuralNetworkModel)
        features: List of feature names
        n_power_lags: Number of power lag features
        hours_test: Hour values for dynamic quantile (optional)
    
    Returns:
        dict: Predictions and metadata
    """
    n_samples = len(X_test)
    predictions = np.zeros(n_samples)
    
    power_lag_indices = [i for i, feat in enumerate(features) if 'power_lag' in feat]
    
    if len(power_lag_indices) == 0:
        predictions = model.predict(X_test, hours_test)
        return {
            'predictions': predictions,
            'iterative': False
        }
    
    for i in range(n_samples):
        X_current = X_test[i:i + 1].copy()
        
        if i > 0:
            for lag_idx, feat_idx in enumerate(power_lag_indices, start=1):
                if i >= lag_idx:
                    X_current[0, feat_idx] = predictions[i - lag_idx]
        
        hour_current = None if hours_test is None else hours_test[i:i + 1]
        predictions[i] = model.predict(X_current, hour_current)[0]
    
    return {
        'predictions': predictions,
        'iterative': True
    }
