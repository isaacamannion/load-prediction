"""
Home Assistant API client for reading history and writing predictions.
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Client for interacting with Home Assistant API."""
    
    def __init__(self, base_url: str, token: str):
        """
        Initialize HA client.
        
        Args:
            base_url: Base URL for HA (e.g., http://supervisor/core)
            token: Supervisor token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def get_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for an entity.
        
        Args:
            entity_id: Entity ID to query
            start_time: Start of history period
            end_time: End of history period (default: now)
        
        Returns:
            List of state dictionaries
        """
        if end_time is None:
            end_time = datetime.now()
        
        # Format timestamps for API
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Build URL
        url = f'{self.base_url}/api/history/period/{start_str}'
        params = {
            'filter_entity_id': entity_id,
            'end_time': end_str,
            'minimal_response': 'true'
        }
        
        logger.info(f"Fetching history for {entity_id} from {start_str} to {end_str}")
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or not isinstance(data, list) or len(data) == 0:
                logger.warning(f"No history data returned for {entity_id}")
                return []
            
            # History returns list of lists, get first element
            entity_history = data[0] if isinstance(data[0], list) else data
            
            logger.info(f"Retrieved {len(entity_history)} history records")
            return entity_history
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch history: {e}")
            raise
    
    def set_state(
        self,
        entity_id: str,
        state: Any,
        attributes: Dict[str, Any] = None
    ) -> bool:
        """
        Set state for an entity.
        
        Args:
            entity_id: Entity ID to update
            state: New state value
            attributes: Optional attributes dict
        
        Returns:
            True if successful
        """
        url = f'{self.base_url}/api/states/{entity_id}'
        
        payload = {
            'state': state,
            'attributes': attributes or {}
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Set state for {entity_id}: {state}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to set state: {e}")
            return False
    
    def create_prediction_sensors(
        self,
        predictions: List[Dict[str, Any]],
        source_entity: str
    ) -> bool:
        """
        Create/update prediction sensors in Home Assistant.
        
        Creates multiple sensors:
        - sensor.power_prediction_next_1h
        - sensor.power_prediction_next_6h
        - sensor.power_prediction_next_12h
        - sensor.power_prediction_next_24h
        - sensor.power_prediction_next_48h
        
        Args:
            predictions: List of prediction dicts with timestamp, actual, predicted
            source_entity: Source entity ID for reference
        
        Returns:
            True if successful
        """
        if not predictions:
            return False
        
        # Calculate average predictions for different time windows
        windows = {
            '1h': 1,
            '6h': 6,
            '12h': 12,
            '24h': 24,
            '48h': 48
        }
        
        success = True
        
        for window_name, hours in windows.items():
            if len(predictions) < hours:
                continue
            
            # Get predictions for this window
            window_predictions = predictions[:hours]
            avg_prediction = sum(p['predicted'] for p in window_predictions) / len(window_predictions)
            max_prediction = max(p['predicted'] for p in window_predictions)
            
            # Create sensor
            entity_id = f'sensor.power_prediction_next_{window_name}'
            
            attributes = {
                'unit_of_measurement': 'kW',
                'friendly_name': f'Power Prediction Next {window_name.upper()}',
                'icon': 'mdi:flash',
                'device_class': 'power',
                'source_entity': source_entity,
                'window': window_name,
                'average': round(avg_prediction, 2),
                'maximum': round(max_prediction, 2),
                'predictions': [
                    {
                        'time': p['timestamp'],
                        'value': round(p['predicted'], 2)
                    }
                    for p in window_predictions
                ]
            }
            
            if not self.set_state(entity_id, round(avg_prediction, 2), attributes):
                success = False
        
        # Also create a sensor with full prediction data
        entity_id = 'sensor.power_prediction_full'
        attributes = {
            'friendly_name': 'Power Prediction Full Dataset',
            'icon': 'mdi:chart-line',
            'source_entity': source_entity,
            'predictions': [
                {
                    'time': p['timestamp'],
                    'predicted': round(p['predicted'], 2),
                    'actual': round(p.get('actual', 0), 2) if p.get('actual') else None
                }
                for p in predictions
            ],
            'count': len(predictions)
        }
        
        self.set_state(entity_id, len(predictions), attributes)
        
        return success
