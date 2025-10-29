#!/usr/bin/env python3
"""
Simple test for data pipeline components.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from pathlib import Path

def test_basic_data_loading():
    """Test basic data loading."""
    print("🧪 Testing basic data loading...")
    
    data_path = "data/processed/merged_sample.csv"
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {df.shape}")
        
        # Basic info
        print(f"   Columns: {df.columns.tolist()[:5]}...")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"   Data types: {df.dtypes.value_counts().to_dict()}")
        
        # Check target column
        if "attack_type" in df.columns:
            print(f"   Target classes: {df['attack_type'].nunique()}")
            print(f"   Target distribution: {df['attack_type'].value_counts().head()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_basic_feature_engineering():
    """Test basic feature engineering."""
    print("\n🧪 Testing basic feature engineering...")
    
    try:
        # Load sample data
        data_path = "data/processed/merged_sample.csv"
        df = pd.read_csv(data_path).head(1000)  # Use smaller sample
        
        print(f"✅ Sample data loaded: {df.shape}")
        
        # Basic feature engineering
        df_features = df.copy()
        
        # Create some simple features
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        
        # Protocol ratios
        if 'TCP' in df_features.columns and 'UDP' in df_features.columns:
            df_features['tcp_udp_ratio'] = df_features['TCP'] / (df_features['UDP'] + 1e-8)
            print("✅ TCP/UDP ratio feature created")
        
        # Statistical features
        if 'Rate' in df_features.columns:
            df_features['rate_log'] = np.log1p(df_features['Rate'])
            df_features['rate_sqrt'] = np.sqrt(df_features['Rate'])
            print("✅ Rate transformation features created")
        
        # Flag features
        flag_cols = [col for col in df_features.columns if 'flag' in col.lower()]
        if flag_cols:
            df_features['flag_sum'] = df_features[flag_cols].sum(axis=1)
            print(f"✅ Flag sum feature created from {len(flag_cols)} flag columns")
        
        print(f"✅ Feature engineering completed: {df_features.shape[1]} total features")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False

def test_data_cleaning():
    """Test data cleaning."""
    print("\n🧪 Testing data cleaning...")
    
    try:
        # Load data
        data_path = "data/processed/merged_sample.csv"
        df = pd.read_csv(data_path).head(1000)
        
        print(f"✅ Data loaded: {df.shape}")
        
        # Check for issues
        print(f"   Infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        
        # Clean data
        df_clean = df.copy()
        
        # Handle infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        print("✅ Infinite values replaced with NaN")
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        print("✅ Missing values filled with median")
        
        # Handle outliers (clip at 99.5 percentile)
        for col in numeric_cols:
            upper = df_clean[col].quantile(0.995)
            df_clean[col] = np.clip(df_clean[col], None, upper)
        print("✅ Outliers clipped at 99.5 percentile")
        
        print(f"✅ Data cleaning completed: {df_clean.shape}")
        print(f"   Missing values after: {df_clean.isnull().sum().sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data cleaning failed: {e}")
        return False

def main():
    """Run basic tests."""
    print("🚀 Starting Basic Data Pipeline Tests...")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Basic Data Loading", test_basic_data_loading),
        ("Basic Feature Engineering", test_basic_feature_engineering),
        ("Data Cleaning", test_data_cleaning)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary:")
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All basic tests passed! Data pipeline foundation is working.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)








