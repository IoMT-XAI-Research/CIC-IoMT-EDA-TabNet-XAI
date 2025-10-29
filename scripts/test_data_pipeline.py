#!/usr/bin/env python3
"""
Test script for the advanced data pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader, load_and_clean_data
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.streaming.stream_processor import StreamProcessor, AlertingStreamProcessor
from src.utils.logging import setup_logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

def test_data_loader():
    """Test DataLoader functionality."""
    print("Testing DataLoader...")
    
    # Initialize loader
    loader = DataLoader()
    
    # Test data loading
    data_path = "data/processed/merged_sample.csv"
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        return False
    
    try:
        df = loader.load_data(data_path)
        print(f"Data loaded: {df.shape}")
        
        # Test data validation
        stats = loader.validate_data(df)
        print(f"Data validation completed")
        print(f"   Missing values: {sum(stats['missing_values'].values())}")
        print(f"   Memory usage: {stats['memory_usage']:.2f} MB")
        
        # Test data cleaning
        df_clean = loader.clean_data(df)
        print(f"Data cleaning completed: {df_clean.shape}")
        
        # Test feature preparation
        X, y = loader.prepare_features(df_clean, target_col="attack_type")
        print(f"✅ Features prepared: {X.shape[1]} features, {len(y)} samples")
        
        # Test data splitting
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
        print(f"Data split completed")
        print(f"   Train: {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        
        # Test feature scaling
        X_train_scaled, X_val_scaled, X_test_scaled = loader.scale_features(X_train, X_val, X_test)
        print(f"Feature scaling completed")
        
        # Test target encoding
        y_train_encoded = loader.encode_target(y_train, fit_encoder=True)
        y_val_encoded = loader.encode_target(y_val, fit_encoder=False)
        y_test_encoded = loader.encode_target(y_test, fit_encoder=False)
        print(f"Target encoding completed")
        
        return True
        
    except Exception as e:
        print(f"DataLoader test failed: {e}")
        return False

def test_feature_engineer():
    """Test FeatureEngineer functionality."""
    print("\nTesting FeatureEngineer...")
    
    try:
        # Load sample data
        data_path = "data/processed/merged_sample.csv"
        if not Path(data_path).exists():
            print(f"Data file not found: {data_path}")
            return False
        
        df = pd.read_csv(data_path).head(1000)  # Use smaller sample for testing
        print(f"Sample data loaded: {df.shape}")
        
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        # Test network features
        df_network = engineer.create_network_features(df)
        print(f"Network features created: {df_network.shape[1] - df.shape[1]} new features")
        
        # Test statistical features
        df_stats = engineer.create_statistical_features(df_network, window_sizes=[5, 10])
        print(f"Statistical features created: {df_stats.shape[1] - df_network.shape[1]} new features")
        
        # Test interaction features
        df_interaction = engineer.create_interaction_features(df_stats)
        print(f"Interaction features created: {df_interaction.shape[1] - df_stats.shape[1]} new features")
        
        # Test polynomial features
        df_poly = engineer.create_polynomial_features(df_interaction, degree=2)
        print(f"Polynomial features created: {df_poly.shape[1] - df_interaction.shape[1]} new features")
        
        # Test comprehensive feature engineering
        if "attack_type" in df.columns:
            X, y = engineer.engineer_all_features(df, target_col="attack_type", 
                                                feature_selection=True, pca=False, scaling=True)
            print(f"Comprehensive feature engineering completed: {X.shape[1]} features")
        else:
            print("No target column found, skipping comprehensive test")
        
        return True
        
    except Exception as e:
        print(f"FeatureEngineer test failed: {e}")
        return False

def test_stream_processor():
    """Test StreamProcessor functionality."""
    print("\nTesting StreamProcessor...")
    
    try:
        # Initialize stream processor
        config = {
            'buffer_size': 100,
            'window_size': 60,
            'batch_size': 50
        }
        processor = StreamProcessor(config)
        
        # Add callback function
        def test_callback(df, predictions):
            print(f"   Callback: {df.shape[0]} data points processed")
            if predictions:
                print(f"   Predictions: {len(predictions.get('predictions', []))} samples")
        
        processor.add_callback(test_callback)
        
        # Test data points
        sample_data = []
        for i in range(10):
            data_point = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'Rate': np.random.uniform(100, 1000),
                'TCP': np.random.choice([0, 1]),
                'UDP': np.random.choice([0, 1]),
                'syn_flag_number': np.random.uniform(0, 1),
                'ack_flag_number': np.random.uniform(0, 1)
            }
            sample_data.append(data_point)
        
        # Add data points
        for data in sample_data:
            processor.add_data_point(data)
        
        print(f"Stream processing test completed")
        
        # Test alerting processor
        alert_config = {
            'buffer_size': 50,
            'window_size': 30,
            'batch_size': 25,
            'alert_threshold': 0.8,
            'alert_cooldown': 60
        }
        alert_processor = AlertingStreamProcessor(alert_config)
        
        def alert_callback(alert_data):
            print(f"   Alert: {alert_data['attack_count']} attacks detected")
        
        alert_processor.add_alert_callback(alert_callback)
        print(f"Alerting stream processor initialized")
        
        return True
        
    except Exception as e:
        print(f"StreamProcessor test failed: {e}")
        return False

def test_integration():
    """Test integrated pipeline."""
    print("\nTesting Integrated Pipeline...")
    
    try:
        # Load and clean data
        data_path = "data/processed/merged_sample.csv"
        if not Path(data_path).exists():
            print(f"Data file not found: {data_path}")
            return False
        
        print("📊 Loading and cleaning data...")
        X, y = load_and_clean_data(data_path, target_col="attack_type")
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Feature engineering
        print("Applying feature engineering...")
        engineer = FeatureEngineer()
        X_engineered, y_engineered = engineer.engineer_all_features(
            pd.concat([X, y], axis=1), 
            target_col="attack_type",
            feature_selection=True,
            pca=False,
            scaling=True
        )
        print(f"Feature engineering completed: {X_engineered.shape[1]} features")
        
        # Data splitting
        print("Splitting data...")
        loader = DataLoader()
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X_engineered, y_engineered)
        print(f"Data split completed")
        
        # Save preprocessors
        print("Saving preprocessors...")
        loader.save_preprocessors("artifacts/preprocessors")
        print("Preprocessors saved")
        
        return True
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting Data Pipeline Tests...")
    print("=" * 50)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Run tests
    tests = [
        ("DataLoader", test_data_loader),
        ("FeatureEngineer", test_feature_engineer),
        ("StreamProcessor", test_stream_processor),
        ("Integration", test_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary:")
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\nAll tests passed! Data pipeline is ready.")
    else:
        print("\nSome tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)








