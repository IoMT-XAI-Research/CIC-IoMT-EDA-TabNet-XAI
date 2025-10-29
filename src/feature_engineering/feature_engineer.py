"""
Advanced feature engineering for IoMT IDS.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for IoMT network traffic data.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.feature_names = None
        self.selected_features = None
        
    def create_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create network-specific features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with network features
        """
        df_features = df.copy()
        logger.info("🌐 Creating network features...")
        
        # Protocol ratios
        protocol_cols = ['HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP']
        available_protocols = [col for col in protocol_cols if col in df.columns]
        
        if available_protocols:
            df_features['protocol_diversity'] = df_features[available_protocols].sum(axis=1)
            df_features['tcp_ratio'] = df_features.get('TCP', 0) / (df_features.get('UDP', 0) + 1e-8)
            df_features['http_ratio'] = df_features.get('HTTP', 0) / (df_features.get('HTTPS', 0) + 1e-8)
        
        # Flag ratios
        flag_cols = ['fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number']
        available_flags = [col for col in flag_cols if col in df.columns]
        
        if available_flags:
            df_features['flag_diversity'] = df_features[available_flags].sum(axis=1)
            df_features['syn_ack_ratio'] = df_features.get('syn_flag_number', 0) / (df_features.get('ack_flag_number', 0) + 1e-8)
            df_features['rst_ratio'] = df_features.get('rst_flag_number', 0) / (df_features.get('syn_flag_number', 0) + 1e-8)
        
        # Statistical features
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_features['packet_rate_std'] = df_features.get('Rate', 0).rolling(window=5, min_periods=1).std()
            df_features['packet_rate_mean'] = df_features.get('Rate', 0).rolling(window=5, min_periods=1).mean()
            df_features['packet_rate_cv'] = df_features['packet_rate_std'] / (df_features['packet_rate_mean'] + 1e-8)
        
        # Time-based features
        if 'IAT' in df.columns:
            df_features['iat_mean'] = df_features['IAT'].rolling(window=5, min_periods=1).mean()
            df_features['iat_std'] = df_features['IAT'].rolling(window=5, min_periods=1).std()
            df_features['iat_cv'] = df_features['iat_std'] / (df_features['iat_mean'] + 1e-8)
        
        # Size-based features
        if 'Tot size' in df.columns:
            df_features['size_efficiency'] = df_features['Tot size'] / (df_features.get('Number', 1) + 1e-8)
            df_features['size_std'] = df_features['Tot size'].rolling(window=5, min_periods=1).std()
        
        logger.info(f"✅ Network features created: {df_features.shape[1] - df.shape[1]} new features")
        return df_features
    
    def create_statistical_features(self, df: pd.DataFrame, 
                                   window_sizes: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create statistical features with rolling windows.
        
        Args:
            df: Input DataFrame
            window_sizes: List of window sizes for rolling statistics
            
        Returns:
            DataFrame with statistical features
        """
        df_features = df.copy()
        logger.info("📊 Creating statistical features...")
        
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        
        for window in window_sizes:
            for col in numeric_cols:
                if col in df_features.columns:
                    # Rolling statistics
                    df_features[f'{col}_mean_{window}'] = df_features[col].rolling(window=window, min_periods=1).mean()
                    df_features[f'{col}_std_{window}'] = df_features[col].rolling(window=window, min_periods=1).std()
                    df_features[f'{col}_min_{window}'] = df_features[col].rolling(window=window, min_periods=1).min()
                    df_features[f'{col}_max_{window}'] = df_features[col].rolling(window=window, min_periods=1).max()
                    
                    # Coefficient of variation
                    df_features[f'{col}_cv_{window}'] = df_features[f'{col}_std_{window}'] / (df_features[f'{col}_mean_{window}'] + 1e-8)
                    
                    # Percentiles
                    df_features[f'{col}_q25_{window}'] = df_features[col].rolling(window=window, min_periods=1).quantile(0.25)
                    df_features[f'{col}_q75_{window}'] = df_features[col].rolling(window=window, min_periods=1).quantile(0.75)
                    df_features[f'{col}_iqr_{window}'] = df_features[f'{col}_q75_{window}'] - df_features[f'{col}_q25_{window}']
        
        logger.info(f"✅ Statistical features created: {df_features.shape[1] - df.shape[1]} new features")
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   feature_pairs: List[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Create interaction features between pairs of features.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of feature pairs for interactions (None for auto-detect)
            
        Returns:
            DataFrame with interaction features
        """
        df_features = df.copy()
        logger.info("🔗 Creating interaction features...")
        
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        
        if feature_pairs is None:
            # Auto-detect important pairs
            feature_pairs = []
            for i, col1 in enumerate(numeric_cols[:10]):  # Limit to first 10 for performance
                for col2 in numeric_cols[i+1:11]:  # Limit interactions
                    if col1 in df_features.columns and col2 in df_features.columns:
                        feature_pairs.append((col1, col2))
        
        for col1, col2 in feature_pairs:
            if col1 in df_features.columns and col2 in df_features.columns:
                # Multiplicative interaction
                df_features[f'{col1}_x_{col2}'] = df_features[col1] * df_features[col2]
                
                # Ratio interaction
                df_features[f'{col1}_div_{col2}'] = df_features[col1] / (df_features[col2] + 1e-8)
                
                # Difference interaction
                df_features[f'{col1}_diff_{col2}'] = df_features[col1] - df_features[col2]
                
                # Sum interaction
                df_features[f'{col1}_sum_{col2}'] = df_features[col1] + df_features[col2]
        
        logger.info(f"✅ Interaction features created: {df_features.shape[1] - df.shape[1]} new features")
        return df_features
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                 degree: int = 2,
                                 include_bias: bool = False) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Args:
            df: Input DataFrame
            degree: Polynomial degree
            include_bias: Whether to include bias term
            
        Returns:
            DataFrame with polynomial features
        """
        df_features = df.copy()
        logger.info(f"📈 Creating polynomial features (degree {degree})...")
        
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Limit to first 5 columns for performance
            if col in df_features.columns:
                for d in range(2, degree + 1):
                    df_features[f'{col}_pow_{d}'] = np.power(df_features[col], d)
        
        logger.info(f"✅ Polynomial features created: {df_features.shape[1] - df.shape[1]} new features")
        return df_features
    
    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time series features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time series features
        """
        df_features = df.copy()
        logger.info("⏰ Creating time series features...")
        
        # Lag features
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Limit to first 5 columns
            if col in df_features.columns:
                for lag in [1, 2, 3, 5]:
                    df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)
        
        # Difference features
        for col in numeric_cols[:5]:
            if col in df_features.columns:
                df_features[f'{col}_diff_1'] = df_features[col].diff(1)
                df_features[f'{col}_diff_2'] = df_features[col].diff(2)
        
        # Moving averages
        for col in numeric_cols[:3]:  # Limit to first 3 columns
            if col in df_features.columns:
                for window in [3, 5, 10]:
                    df_features[f'{col}_ma_{window}'] = df_features[col].rolling(window=window, min_periods=1).mean()
        
        logger.info(f"✅ Time series features created: {df_features.shape[1] - df.shape[1]} new features")
        return df_features
    
    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       method: str = "mutual_info",
                       k: int = 50,
                       random_state: int = 42) -> pd.DataFrame:
        """
        Select best features using various methods.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Selection method (mutual_info, f_classif, variance)
            k: Number of features to select
            random_state: Random state
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"🎯 Selecting features using {method} (k={k})...")
        
        if method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == "f_classif":
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == "variance":
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Fit selector
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        if hasattr(selector, 'get_support'):
            selected_mask = selector.get_support()
            selected_features = X.columns[selected_mask].tolist()
        else:
            selected_features = X.columns.tolist()
        
        # Create DataFrame with selected features
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        self.feature_selector = selector
        self.selected_features = selected_features
        
        logger.info(f"✅ Feature selection completed: {len(selected_features)} features selected")
        return X_selected_df
    
    def apply_pca(self, X: pd.DataFrame, 
                  n_components: int = 0.95,
                  random_state: int = 42) -> pd.DataFrame:
        """
        Apply PCA dimensionality reduction.
        
        Args:
            X: Feature DataFrame
            n_components: Number of components (int) or variance ratio (float)
            random_state: Random state
            
        Returns:
            DataFrame with PCA features
        """
        logger.info(f"🔍 Applying PCA (n_components={n_components})...")
        
        self.pca = PCA(n_components=n_components, random_state=random_state)
        X_pca = self.pca.fit_transform(X)
        
        # Create feature names
        n_components_actual = X_pca.shape[1]
        pca_columns = [f'PC_{i+1}' for i in range(n_components_actual)]
        
        X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        logger.info(f"✅ PCA completed: {n_components_actual} components, "
                   f"explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return X_pca_df
    
    def scale_features(self, X: pd.DataFrame, 
                      method: str = "standard",
                      fit_scaler: bool = True) -> pd.DataFrame:
        """
        Scale features using various scaling methods.
        
        Args:
            X: Feature DataFrame
            method: Scaling method (standard, minmax, robust)
            fit_scaler: Whether to fit the scaler
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"⚖️ Scaling features using {method}...")
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if fit_scaler:
            X_scaled = scaler.fit_transform(X)
            self.scaler = scaler
        else:
            X_scaled = scaler.transform(X)
        
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        logger.info("✅ Features scaled successfully")
        return X_scaled_df
    
    def engineer_all_features(self, df: pd.DataFrame, 
                             target_col: str = "attack_type",
                             feature_selection: bool = True,
                             pca: bool = False,
                             scaling: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            feature_selection: Whether to apply feature selection
            pca: Whether to apply PCA
            scaling: Whether to apply scaling
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("🚀 Starting comprehensive feature engineering...")
        
        # Step 1: Network features
        df_features = self.create_network_features(df)
        
        # Step 2: Statistical features
        df_features = self.create_statistical_features(df_features)
        
        # Step 3: Interaction features
        df_features = self.create_interaction_features(df_features)
        
        # Step 4: Polynomial features
        df_features = self.create_polynomial_features(df_features, degree=2)
        
        # Step 5: Time series features
        df_features = self.create_time_series_features(df_features)
        
        # Separate features and target
        if target_col in df_features.columns:
            X = df_features.drop(columns=[target_col])
            y = df_features[target_col]
        else:
            X = df_features
            y = None
        
        # Step 6: Feature selection
        if feature_selection and y is not None:
            X = self.select_features(X, y, method="mutual_info", k=100)
        
        # Step 7: PCA
        if pca:
            X = self.apply_pca(X, n_components=0.95)
        
        # Step 8: Scaling
        if scaling:
            X = self.scale_features(X, method="standard", fit_scaler=True)
        
        logger.info(f"✅ Feature engineering completed: {X.shape[1]} features, {X.shape[0]} samples")
        
        return X, y if y is not None else None








