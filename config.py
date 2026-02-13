"""
Configuration classes for energy load prediction.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DataConfig:
    """Configuration for data processing."""
    csv_file: str
    bin_size_minutes: int = 60
    n_power_lags: int = 10
    n_temp_lags: int = 5
    train_percentage: float = 80.0
    
    def __post_init__(self):
        if not (10 <= self.train_percentage <= 95):
            raise ValueError("train_percentage must be between 10 and 95")


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: str = 'quantile'  # 'quantile' or 'neural' or 'both'
    quantile: float = 0.75
    use_dynamic_quantile: bool = False
    peak_start: int = 9
    peak_end: int = 22
    peak_quantile: float = 0.75
    offpeak_quantile: float = 0.50
    
    def __post_init__(self):
        if self.model_type not in ['quantile', 'neural', 'both']:
            raise ValueError("model_type must be 'quantile', 'neural', or 'both'")
        if not (0.5 <= self.quantile <= 0.99):
            raise ValueError("quantile must be between 0.5 and 0.99")
    
    def get_dynamic_config(self):
        """Get dynamic quantile configuration if enabled."""
        if self.use_dynamic_quantile:
            return {
                'peak_start': self.peak_start,
                'peak_end': self.peak_end,
                'peak_quantile': self.peak_quantile,
                'offpeak_quantile': self.offpeak_quantile
            }
        return None


@dataclass
class PostProcessConfig:
    """Configuration for post-processing."""
    buffer_percentage: float = 0.0
    apply_smoothing: bool = False
    smoothing_method: str = 'moving_average'  # 'moving_average' or 'exponential'
    smoothing_window: int = 3
    
    def __post_init__(self):
        if not (0 <= self.buffer_percentage <= 50):
            raise ValueError("buffer_percentage must be between 0 and 50")
        if self.smoothing_method not in ['moving_average', 'exponential']:
            raise ValueError("smoothing_method must be 'moving_average' or 'exponential'")


def get_recommended_config():
    """
    Get recommended configuration for good results.
    
    Returns:
        tuple: (DataConfig, ModelConfig, PostProcessConfig)
    """
    data_config = DataConfig(
        csv_file='HA_history_export.csv',
        bin_size_minutes=60,
        n_power_lags=10,
        n_temp_lags=5,
        train_percentage=80.0
    )
    
    model_config = ModelConfig(
        model_type='quantile',
        quantile=0.75,
        use_dynamic_quantile=True,
        peak_start=9,
        peak_end=22,
        peak_quantile=0.75,
        offpeak_quantile=0.50
    )
    
    post_config = PostProcessConfig(
        buffer_percentage=0.0,
        apply_smoothing=False
    )
    
    return data_config, model_config, post_config
