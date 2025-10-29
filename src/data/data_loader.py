"""
Advanced data loading and preprocessing for IoMT IDS.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Advanced data loader for IoMT IDS dataset.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize DataLoader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_names = None
        
    def load_data(self, data_path: str, file_format: str = "auto") -> pd.DataFrame:
        """
        Load data from various formats.
        
        Args:
            data_path: Path to data file
            file_format: File format (auto, csv, parquet, json)
            
        Returns:
            Loaded DataFrame
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Auto-detect format if not specified
        if file_format == "auto":
            if data_path.suffix == ".csv":
                file_format = "csv"
            elif data_path.suffix == ".parquet":
                file_format = "parquet"
            elif data_path.suffix == ".json":
                file_format = "json"
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        logger.info(f"Loading data from {data_path} ({file_format})")
        
        try:
            if file_format == "csv":
                df = pd.read_csv(data_path)
            elif file_format == "parquet":
                df = pd.read_parquet(data_path)
            elif file_format == "json":
                df = pd.read_json(data_path)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
                
            logger.info(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data quality and return statistics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation statistics
        """
        stats = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'duplicates': df.duplicated().sum(),
            'infinite_values': np.isinf(df.select_dtypes(include=[np.number])).sum().to_dict()
        }
        
        logger.info(f"📊 Data validation completed:")
        logger.info(f"   Shape: {stats['shape']}")
        logger.info(f"   Missing values: {sum(stats['missing_values'].values())}")
        logger.info(f"   Memory usage: {stats['memory_usage']:.2f} MB")
        logger.info(f"   Duplicates: {stats['duplicates']}")
        
        return stats
    
    def clean_data(self, df: pd.DataFrame, 
                   handle_inf: str = "replace",
                   handle_missing: str = "median",
                   handle_outliers: str = "clip",
                   outlier_threshold: float = 0.995) -> pd.DataFrame:
        """
        Clean data with various strategies.
        
        Args:
            df: DataFrame to clean
            handle_inf: How to handle infinite values (replace, drop, keep)
            handle_missing: How to handle missing values (median, mean, drop, forward_fill)
            handle_outliers: How to handle outliers (clip, remove, keep)
            outlier_threshold: Threshold for outlier detection
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        logger.info("🧹 Starting data cleaning...")
        
        # Handle infinite values
        if handle_inf == "replace":
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
            logger.info("   ✅ Replaced infinite values with NaN")
        elif handle_inf == "drop":
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
            df_clean = df_clean.dropna()
            logger.info("   ✅ Dropped rows with infinite values")
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if handle_missing == "median":
            for col in numeric_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            logger.info("   ✅ Filled missing values with median")
        elif handle_missing == "mean":
            for col in numeric_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            logger.info("   ✅ Filled missing values with mean")
        elif handle_missing == "forward_fill":
            df_clean = df_clean.fillna(method='ffill')
            logger.info("   ✅ Forward filled missing values")
        elif handle_missing == "drop":
            df_clean = df_clean.dropna()
            logger.info("   ✅ Dropped rows with missing values")
        
        # Handle outliers
        if handle_outliers == "clip":
            for col in numeric_cols:
                upper = df_clean[col].quantile(outlier_threshold)
                df_clean[col] = np.clip(df_clean[col], None, upper)
            logger.info(f"   ✅ Clipped outliers at {outlier_threshold} percentile")
        elif handle_outliers == "remove":
            for col in numeric_cols:
                upper = df_clean[col].quantile(outlier_threshold)
                df_clean = df_clean[df_clean[col] <= upper]
            logger.info(f"   ✅ Removed outliers above {outlier_threshold} percentile")
        
        logger.info(f"✅ Data cleaning completed: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str = "attack_type",
                        feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.
        
        Args:
            df: DataFrame with data
            target_col: Name of target column
            feature_cols: List of feature columns (None for auto-detect)
            
        Returns:
            Tuple of (features_df, target_series)
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        # Separate features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Store feature names
        self.feature_names = feature_cols
        
        logger.info(f"📊 Features prepared: {X.shape[1]} features, {len(y)} samples")
        logger.info(f"   Target classes: {y.nunique()}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = 0.2,
                   val_size: float = 0.2,
                   random_state: int = 42,
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                  pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Test set size
            val_size: Validation set size (of remaining data)
            random_state: Random state for reproducibility
            stratify: Whether to stratify by target
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        stratify_param = y if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        # Second split: train vs val
        if stratify:
            stratify_param = y_temp
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=stratify_param
        )
        
        logger.info(f"📊 Data split completed:")
        logger.info(f"   Train: {X_train.shape[0]} samples")
        logger.info(f"   Validation: {X_val.shape[0]} samples")
        logger.info(f"   Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train: pd.DataFrame, 
                      X_val: pd.DataFrame = None,
                      X_test: pd.DataFrame = None,
                      fit_scaler: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            fit_scaler: Whether to fit the scaler on training data
            
        Returns:
            Tuple of scaled features
        """
        if fit_scaler:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            logger.info("✅ Scaler fitted on training data")
        else:
            X_train_scaled = pd.DataFrame(
                self.scaler.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        
        X_val_scaled = None
        X_test_scaled = None
        
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        logger.info("✅ Features scaled successfully")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def encode_target(self, y: pd.Series, fit_encoder: bool = True) -> pd.Series:
        """
        Encode target labels.
        
        Args:
            y: Target Series
            fit_encoder: Whether to fit the encoder
            
        Returns:
            Encoded target Series
        """
        if fit_encoder:
            y_encoded = pd.Series(
                self.label_encoder.fit_transform(y),
                index=y.index,
                name=y.name
            )
            self.target_names = self.label_encoder.classes_
            logger.info(f"✅ Target encoded: {len(self.target_names)} classes")
        else:
            y_encoded = pd.Series(
                self.label_encoder.transform(y),
                index=y.index,
                name=y.name
            )
        
        return y_encoded
    
    def save_preprocessors(self, save_path: str) -> None:
        """
        Save fitted preprocessors.
        
        Args:
            save_path: Path to save preprocessors
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, save_path / "scaler.pkl")
        
        # Save label encoder
        joblib.dump(self.label_encoder, save_path / "label_encoder.pkl")
        
        # Save feature names
        joblib.dump(self.feature_names, save_path / "feature_names.pkl")
        
        logger.info(f"✅ Preprocessors saved to {save_path}")
    
    def load_preprocessors(self, load_path: str) -> None:
        """
        Load fitted preprocessors.
        
        Args:
            load_path: Path to load preprocessors from
        """
        load_path = Path(load_path)
        
        # Load scaler
        self.scaler = joblib.load(load_path / "scaler.pkl")
        
        # Load label encoder
        self.label_encoder = joblib.load(load_path / "label_encoder.pkl")
        
        # Load feature names
        self.feature_names = joblib.load(load_path / "feature_names.pkl")
        
        logger.info(f"✅ Preprocessors loaded from {load_path}")


def load_and_clean_data(data_path: str, 
                        target_col: str = "attack_type",
                        output_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load and clean data.
    
    Args:
        data_path: Path to data file
        target_col: Name of target column
        output_path: Path to save cleaned data (optional)
        
    Returns:
        Tuple of (features_df, target_series)
    """
    loader = DataLoader()
    
    # Load data
    df = loader.load_data(data_path)
    
    # Validate data
    stats = loader.validate_data(df)
    
    # Clean data
    df_clean = loader.clean_data(df)
    
    # Prepare features
    X, y = loader.prepare_features(df_clean, target_col)
    
    # Save cleaned data if path provided
    if output_path:
        df_clean.to_parquet(output_path, index=False)
        logger.info(f"💾 Cleaned data saved to {output_path}")
    
    return X, y








