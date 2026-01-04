"""
train_test_run_5.py
========================
MASTER PIPELINE GENERATOR for IoMT IDS
- NO SMOTE (Class Weights Only)
- Exports exact column structure for inference
- Saves all required artifacts for simulate_traffic.py

CRITICAL ARTIFACTS SAVED:
1. final_feature_names.pkl - Column list after One-Hot Encoding
2. scaler.pkl - StandardScaler fitted on X_train
3. label_encoder_binary.pkl
4. label_encoder_multiclass.pkl
5. binary_model.zip / multiclass_model.zip
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.loader import DataLoader
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import FloatType
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier

try:
    import shap
except ImportError:
    shap = None


class MasterPipeline:
    """
    Master Pipeline Generator for IoMT IDS.
    - NO SMOTE: Uses class weights for imbalance
    - Exports exact column structure for inference scripts
    """
    
    ALL_CLASSES = ['Benign', 'DDoS', 'DoS', 'MQTT', 'Recon', 'Spoofing']
    
    def __init__(self, base_path, artifacts_dir="artifacts"):
        self.base_path = base_path
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoder_binary = LabelEncoder()
        self.label_encoder_multi = LabelEncoder()
        
        # Pre-fit encoders with known classes
        self.label_encoder_binary.fit(['Benign', 'Attack'])
        self.label_encoder_multi.fit(self.ALL_CLASSES)
        
        # Will be populated during pipeline
        self.final_feature_names = None
        self.categorical_columns = []
    
    # =========================================================================
    # STEP 1: LOAD & CLEAN
    # =========================================================================
    def load_and_clean(self, fraction=0.3):
        print("\n" + "="*60)
        print("[STEP 1] Loading & Cleaning Data")
        print("="*60)
        
        loader = DataLoader(self.base_path)
        df_spark = loader.load_data()
        if df_spark is None:
            sys.exit(1)
        
        # Cast Rate/Variance
        for c in ['Rate', 'Variance']:
            if c in df_spark.columns:
                df_spark = df_spark.withColumn(c, col(c).cast(FloatType()))
        
        # Sanitize Infinity
        for f in df_spark.schema.fields:
            if str(f.dataType) in ['FloatType()', 'DoubleType()']:
                df_spark = df_spark.withColumn(f.name, 
                    when(col(f.name) == float("inf"), lit(None))
                    .when(col(f.name) == float("-inf"), lit(None))
                    .otherwise(col(f.name)))
        
        df = df_spark.sample(False, fraction, 42).toPandas()
        df['Label'] = df['Label'].replace('', 'MQTT').fillna('MQTT')
        
        # Drop leakage columns
        df.columns = df.columns.str.strip()
        leakage_cols = ['Source IP', 'Destination IP', 'Timestamp', 'Flow ID', 
                        'Source Port', 'Destination Port', 'Unnamed: 0', 
                        'source_file', 'basename', 'SubType', 'Src IP', 'Dst IP', 
                        'MAC', 'Prob', 'Flow Duration', 'Time', 'Protocol']
        df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
        
        # Mode imputation
        for c in df.columns:
            if df[c].isnull().any():
                mode_val = df[c].mode()
                df[c] = df[c].fillna(mode_val[0] if len(mode_val) > 0 else 0)
        
        print(f"  - Shape: {df.shape}")
        print(f"  - Labels: {df['Label'].value_counts().to_dict()}")
        return df
    
    # =========================================================================
    # STEP 2: ONE-HOT ENCODING
    # =========================================================================
    def one_hot_encode(self, df):
        print("\n" + "="*60)
        print("[STEP 2] One-Hot Encoding")
        print("="*60)
        
        X = df.drop(columns=['Label'])
        y_raw = df['Label']
        
        # Identify categorical columns
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"  - Categorical columns: {self.categorical_columns}")
        
        # Drop high-cardinality
        for c in list(self.categorical_columns):
            if X[c].nunique() > 50:
                print(f"    [DROP] {c} (cardinality: {X[c].nunique()})")
                X = X.drop(columns=[c])
                self.categorical_columns.remove(c)
        
        # Apply One-Hot Encoding
        if self.categorical_columns:
            X = pd.get_dummies(X, columns=self.categorical_columns, dummy_na=True)
        
        X = X.astype(float)
        
        # =====================================================
        # CRITICAL: Save feature names AFTER One-Hot Encoding
        # =====================================================
        self.final_feature_names = X.columns.tolist()
        
        print(f"  - Final feature count: {len(self.final_feature_names)}")
        print(f"  - First 10 features: {self.final_feature_names[:10]}")
        
        # Create targets
        y_binary = (y_raw != 'Benign').astype(int).values
        y_multi = self.label_encoder_multi.transform(y_raw)
        
        return X, y_binary, y_multi
    
    # =========================================================================
    # STEP 3: SPLIT DATA
    # =========================================================================
    def split_data(self, X, y_binary, y_multi):
        print("\n" + "="*60)
        print("[STEP 3] Stratified Split (70/15/15)")
        print("="*60)
        
        X_train, X_temp, y_b_train, y_b_temp, y_m_train, y_m_temp = train_test_split(
            X, y_binary, y_multi, 
            test_size=0.30, stratify=y_multi, random_state=42
        )
        
        X_val, X_test, y_b_val, y_b_test, y_m_val, y_m_test = train_test_split(
            X_temp, y_b_temp, y_m_temp, 
            test_size=0.50, stratify=y_m_temp, random_state=42
        )
        
        print(f"  - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return (X_train, X_val, X_test, 
                y_b_train, y_b_val, y_b_test,
                y_m_train, y_m_val, y_m_test)
    
    # =========================================================================
    # STEP 4: ALIGN FEATURES (Ensure same columns across splits)
    # =========================================================================
    def align_features(self, X_train, X_val, X_test):
        print("\n" + "="*60)
        print("[STEP 4] Aligning Features")
        print("="*60)
        
        train_cols = set(X_train.columns)
        
        for name, X in [('Val', X_val), ('Test', X_test)]:
            # Add missing columns
            missing = train_cols - set(X.columns)
            for col in missing:
                X[col] = 0
            # Remove extra columns
            extra = set(X.columns) - train_cols
            if extra:
                X.drop(columns=list(extra), inplace=True)
        
        # Ensure same column order
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]
        
        print(f"  - All splits aligned to {len(X_train.columns)} columns")
        
        return X_train, X_val, X_test
    
    # =========================================================================
    # STEP 5: SCALE (Fit on Train Only)
    # =========================================================================
    def scale_data(self, X_train, X_val, X_test):
        print("\n" + "="*60)
        print("[STEP 5] Scaling (Fit on Train Only)")
        print("="*60)
        
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        X_test_s = self.scaler.transform(X_test)
        
        print("  - Scaler fitted on training data only.")
        
        return X_train_s, X_val_s, X_test_s
    
    # =========================================================================
    # STEP 6: COMPUTE CLASS WEIGHTS (NO SMOTE)
    # =========================================================================
    def compute_weights(self, y_train, name):
        print(f"\n  [Class Weights for {name}]")
        
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes.astype(int), weights))
        
        print(f"    Classes: {classes}")
        print(f"    Weights: {weight_dict}")
        
        return weight_dict
    
    # =========================================================================
    # STEP 7: TRAIN TABNET
    # =========================================================================
    def train_tabnet(self, X_train, y_train, X_val, y_val, class_weights, name):
        print(f"\n{'='*60}")
        print(f"[TRAINING] {name}")
        print("="*60)
        
        clf = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            verbose=1
        )
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_name=['train', 'valid'],
            eval_metric=['balanced_accuracy'],
            max_epochs=50,
            patience=10,
            batch_size=16384,
            virtual_batch_size=1024,
            num_workers=0,
            drop_last=False,
            weights=class_weights
        )
        
        clf.save_model(os.path.join(self.artifacts_dir, name))
        print(f"  - Saved {name}.zip")
        
        return clf
    
    # =========================================================================
    # STEP 8: EVALUATE
    # =========================================================================
    def evaluate(self, clf, X_test, y_test, name, target_names):
        print(f"\n{'='*60}")
        print(f"[EVALUATION] {name}")
        print("="*60)
        
        y_pred = clf.predict(X_test)
        
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        print(report)
        
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        print(f"  >> Balanced Accuracy: {bal_acc:.4f}")
        
        with open(os.path.join(self.artifacts_dir, f"{name}_report.txt"), "w") as f:
            f.write(report)
            f.write(f"\nBalanced Accuracy: {bal_acc:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.artifacts_dir, f"cm_{name}.png"))
        plt.close()
        
        return bal_acc
    
    # =========================================================================
    # SAVE ALL ARTIFACTS
    # =========================================================================
    def save_artifacts(self):
        print("\n" + "="*60)
        print("[SAVING ARTIFACTS] (For simulate_traffic.py)")
        print("="*60)
        
        # 1. Feature Names (CRITICAL for inference)
        joblib.dump(self.final_feature_names, 
                    os.path.join(self.artifacts_dir, "final_feature_names.pkl"))
        print(f"  ✓ final_feature_names.pkl ({len(self.final_feature_names)} columns)")
        
        # 2. Scaler
        joblib.dump(self.scaler, 
                    os.path.join(self.artifacts_dir, "scaler.pkl"))
        print("  ✓ scaler.pkl")
        
        # 2b. Save as final_preprocessor.pkl (CRITICAL for simulation)
        joblib.dump(self.scaler, 
                    os.path.join(self.artifacts_dir, "final_preprocessor.pkl"))
        print("  ✓ final_preprocessor.pkl (same as scaler)")
        
        # 3. Label Encoders
        joblib.dump(self.label_encoder_binary, 
                    os.path.join(self.artifacts_dir, "label_encoder_binary.pkl"))
        print("  ✓ label_encoder_binary.pkl")
        
        joblib.dump(self.label_encoder_multi, 
                    os.path.join(self.artifacts_dir, "label_encoder_multiclass.pkl"))
        print("  ✓ label_encoder_multiclass.pkl")
        
        # 4. Categorical columns (for recreating One-Hot)
        joblib.dump(self.categorical_columns, 
                    os.path.join(self.artifacts_dir, "categorical_columns.pkl"))
        print(f"  ✓ categorical_columns.pkl ({self.categorical_columns})")
        
        # Print column verification
        print("\n" + "-"*60)
        print("COLUMN MAP FOR INFERENCE:")
        print("-"*60)
        print(f"Total Columns: {len(self.final_feature_names)}")
        print(f"First 20: {self.final_feature_names[:20]}")
        print(f"Last 10: {self.final_feature_names[-10:]}")
    
    # =========================================================================
    # MAIN RUN
    # =========================================================================
    def run(self):
        # 1. Load
        df = self.load_and_clean(fraction=0.3)
        
        # 2. One-Hot Encode (BEFORE split)
        X, y_binary, y_multi = self.one_hot_encode(df)
        
        # 3. Split
        (X_train, X_val, X_test, 
         y_b_train, y_b_val, y_b_test,
         y_m_train, y_m_val, y_m_test) = self.split_data(X, y_binary, y_multi)
        
        # 4. Align Features
        X_train, X_val, X_test = self.align_features(X_train, X_val, X_test)
        
        # Update final feature names after alignment
        self.final_feature_names = X_train.columns.tolist()
        
        # 5. Scale (Fit Train Only)
        X_train_s, X_val_s, X_test_s = self.scale_data(X_train, X_val, X_test)
        
        # 6. Compute Weights
        print("\n" + "="*60)
        print("[STEP 6] Computing Class Weights (NO SMOTE)")
        print("="*60)
        weights_binary = self.compute_weights(y_b_train, "Binary")
        weights_multi = self.compute_weights(y_m_train, "Multiclass")
        
        # 7. Train Binary Model
        clf_binary = self.train_tabnet(
            X_train_s, y_b_train, 
            X_val_s, y_b_val, 
            weights_binary, "binary_model"
        )
        
        # 8. Train Multiclass Model
        clf_multi = self.train_tabnet(
            X_train_s, y_m_train, 
            X_val_s, y_m_val, 
            weights_multi, "multiclass_model"
        )
        
        # 9. Evaluate
        self.evaluate(clf_binary, X_test_s, y_b_test, "Binary", ['Benign', 'Attack'])
        self.evaluate(clf_multi, X_test_s, y_m_test, "Multiclass", self.ALL_CLASSES)
        
        # 10. Save All Artifacts
        self.save_artifacts()
        
        print("\n" + "="*60)
        print("[DONE] Master Pipeline Complete!")
        print("="*60)
        print("\nTo use in simulate_traffic.py:")
        print("  1. Load 'final_feature_names.pkl' to know expected columns")
        print("  2. Create DataFrame with those exact columns")
        print("  3. Apply 'scaler.pkl' to transform")
        print("  4. Run model.predict()")


if __name__ == "__main__":
    MasterPipeline(base_path="data/raw/WiFi_and_MQTT").run()
