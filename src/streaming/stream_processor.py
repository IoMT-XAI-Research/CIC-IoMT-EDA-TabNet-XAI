"""
Real-time stream processing for IoMT IDS.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from datetime import datetime, timedelta
import json
import asyncio
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Real-time stream processor for IoMT network traffic.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize StreamProcessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.buffer = deque(maxlen=self.config.get('buffer_size', 1000))
        self.window_size = self.config.get('window_size', 60)  # seconds
        self.batch_size = self.config.get('batch_size', 100)
        self.is_processing = False
        self.callbacks = []
        self.feature_engineer = None
        self.model = None
        
    def add_callback(self, callback: Callable) -> None:
        """
        Add callback function for processed data.
        
        Args:
            callback: Callback function
        """
        self.callbacks.append(callback)
        logger.info(f"✅ Callback added: {callback.__name__}")
    
    def remove_callback(self, callback: Callable) -> None:
        """
        Remove callback function.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"✅ Callback removed: {callback.__name__}")
    
    def add_data_point(self, data: Dict) -> None:
        """
        Add single data point to stream buffer.
        
        Args:
            data: Data point dictionary
        """
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        
        # Add to buffer
        self.buffer.append(data)
        
        # Process if buffer is full
        if len(self.buffer) >= self.batch_size:
            self._process_batch()
    
    def add_batch(self, data_batch: List[Dict]) -> None:
        """
        Add batch of data points to stream buffer.
        
        Args:
            data_batch: List of data point dictionaries
        """
        for data in data_batch:
            self.add_data_point(data)
    
    def _process_batch(self) -> None:
        """
        Process current batch of data.
        """
        if not self.buffer:
            return
        
        logger.info(f"🔄 Processing batch: {len(self.buffer)} data points")
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.buffer))
        
        # Process features if feature engineer is available
        if self.feature_engineer:
            try:
                df_processed = self._engineer_features(df)
            except Exception as e:
                logger.error(f"❌ Feature engineering failed: {e}")
                df_processed = df
        else:
            df_processed = df
        
        # Make predictions if model is available
        predictions = None
        if self.model:
            try:
                predictions = self._make_predictions(df_processed)
            except Exception as e:
                logger.error(f"❌ Prediction failed: {e}")
        
        # Call callbacks
        for callback in self.callbacks:
            try:
                callback(df_processed, predictions)
            except Exception as e:
                logger.error(f"❌ Callback failed: {e}")
        
        # Clear buffer
        self.buffer.clear()
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for stream data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        if not self.feature_engineer:
            return df
        
        # Apply feature engineering
        df_features = self.feature_engineer.create_network_features(df)
        df_features = self.feature_engineer.create_statistical_features(df_features, window_sizes=[5, 10])
        
        return df_features
    
    def _make_predictions(self, df: pd.DataFrame) -> Dict:
        """
        Make predictions using the model.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Dictionary with predictions
        """
        if not self.model:
            return None
        
        try:
            # Prepare features for prediction
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'attack_type']]
            X = df[feature_cols]
            
            # Make predictions
            predictions = self.model.predict(X.values)
            probabilities = self.model.predict_proba(X.values) if hasattr(self.model, 'predict_proba') else None
            
            # Create results
            results = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'confidence': np.max(probabilities, axis=1).tolist() if probabilities is not None else None,
                'timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return None
    
    def set_feature_engineer(self, feature_engineer) -> None:
        """
        Set feature engineer for stream processing.
        
        Args:
            feature_engineer: FeatureEngineer instance
        """
        self.feature_engineer = feature_engineer
        logger.info("✅ Feature engineer set")
    
    def set_model(self, model) -> None:
        """
        Set model for predictions.
        
        Args:
            model: Trained model instance
        """
        self.model = model
        logger.info("✅ Model set")
    
    def start_processing(self) -> None:
        """
        Start stream processing.
        """
        self.is_processing = True
        logger.info("🚀 Stream processing started")
    
    def stop_processing(self) -> None:
        """
        Stop stream processing.
        """
        self.is_processing = False
        logger.info("⏹️ Stream processing stopped")
    
    def get_buffer_status(self) -> Dict:
        """
        Get current buffer status.
        
        Returns:
            Dictionary with buffer status
        """
        return {
            'buffer_size': len(self.buffer),
            'max_buffer_size': self.buffer.maxlen,
            'is_processing': self.is_processing,
            'window_size': self.window_size,
            'batch_size': self.batch_size
        }


class WindowedStreamProcessor(StreamProcessor):
    """
    Windowed stream processor with time-based windows.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize WindowedStreamProcessor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.windows = {}
        self.window_duration = self.config.get('window_duration', 60)  # seconds
        self.slide_interval = self.config.get('slide_interval', 10)  # seconds
        
    def add_data_point(self, data: Dict) -> None:
        """
        Add data point to appropriate time windows.
        
        Args:
            data: Data point dictionary
        """
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        
        timestamp = datetime.fromisoformat(data['timestamp'])
        
        # Add to relevant windows
        window_start = timestamp.replace(second=0, microsecond=0)
        for i in range(0, self.window_duration, self.slide_interval):
            window_key = window_start + timedelta(seconds=i)
            if window_key not in self.windows:
                self.windows[window_key] = deque(maxlen=self.batch_size)
            
            self.windows[window_key].append(data)
        
        # Process expired windows
        self._process_expired_windows()
    
    def _process_expired_windows(self) -> None:
        """
        Process windows that have expired.
        """
        current_time = datetime.now()
        expired_windows = []
        
        for window_key, window_data in self.windows.items():
            if current_time - window_key > timedelta(seconds=self.window_duration):
                expired_windows.append(window_key)
                self._process_window(window_key, window_data)
        
        # Remove expired windows
        for window_key in expired_windows:
            del self.windows[window_key]
    
    def _process_window(self, window_key: datetime, window_data: deque) -> None:
        """
        Process a single time window.
        
        Args:
            window_key: Window timestamp
            window_data: Data in the window
        """
        if not window_data:
            return
        
        logger.info(f"🪟 Processing window: {window_key} ({len(window_data)} data points)")
        
        # Convert to DataFrame
        df = pd.DataFrame(list(window_data))
        
        # Process features
        if self.feature_engineer:
            try:
                df_processed = self._engineer_features(df)
            except Exception as e:
                logger.error(f"❌ Feature engineering failed: {e}")
                df_processed = df
        else:
            df_processed = df
        
        # Make predictions
        predictions = None
        if self.model:
            try:
                predictions = self._make_predictions(df_processed)
            except Exception as e:
                logger.error(f"❌ Prediction failed: {e}")
        
        # Call callbacks
        for callback in self.callbacks:
            try:
                callback(df_processed, predictions, window_key)
            except Exception as e:
                logger.error(f"❌ Callback failed: {e}")


class AlertingStreamProcessor(WindowedStreamProcessor):
    """
    Stream processor with alerting capabilities.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize AlertingStreamProcessor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.alert_threshold = self.config.get('alert_threshold', 0.8)
        self.alert_cooldown = self.config.get('alert_cooldown', 300)  # seconds
        self.last_alert_time = {}
        self.alert_callbacks = []
    
    def add_alert_callback(self, callback: Callable) -> None:
        """
        Add alert callback function.
        
        Args:
            callback: Alert callback function
        """
        self.alert_callbacks.append(callback)
        logger.info(f"✅ Alert callback added: {callback.__name__}")
    
    def _process_window(self, window_key: datetime, window_data: deque) -> None:
        """
        Process window with alerting.
        
        Args:
            window_key: Window timestamp
            window_data: Data in the window
        """
        # Call parent method
        super()._process_window(window_key, window_data)
        
        # Check for alerts
        self._check_alerts(window_key, window_data)
    
    def _check_alerts(self, window_key: datetime, window_data: deque) -> None:
        """
        Check for alert conditions.
        
        Args:
            window_key: Window timestamp
            window_data: Data in the window
        """
        if not self.model:
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(list(window_data))
            
            # Make predictions
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'attack_type']]
            X = df[feature_cols]
            
            predictions = self.model.predict(X.values)
            probabilities = self.model.predict_proba(X.values) if hasattr(self.model, 'predict_proba') else None
            
            if probabilities is not None:
                # Check for high-confidence attack predictions
                max_probs = np.max(probabilities, axis=1)
                attack_indices = np.where(max_probs > self.alert_threshold)[0]
                
                if len(attack_indices) > 0:
                    # Check cooldown
                    current_time = datetime.now()
                    if window_key not in self.last_alert_time or \
                       (current_time - self.last_alert_time[window_key]).seconds > self.alert_cooldown:
                        
                        # Generate alert
                        alert_data = {
                            'window_key': window_key.isoformat(),
                            'attack_count': len(attack_indices),
                            'max_confidence': float(np.max(max_probs)),
                            'attack_types': predictions[attack_indices].tolist(),
                            'timestamp': current_time.isoformat()
                        }
                        
                        # Send alert
                        self._send_alert(alert_data)
                        self.last_alert_time[window_key] = current_time
                        
        except Exception as e:
            logger.error(f"❌ Alert checking failed: {e}")
    
    def _send_alert(self, alert_data: Dict) -> None:
        """
        Send alert to all alert callbacks.
        
        Args:
            alert_data: Alert data dictionary
        """
        logger.warning(f"🚨 ALERT: {alert_data['attack_count']} attacks detected "
                       f"(confidence: {alert_data['max_confidence']:.3f})")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"❌ Alert callback failed: {e}")








