"""
train_test_run_5.py
========================
Final Production Pipeline for IoMT IDS
- Dual-Model Architecture (Binary + Multiclass)
- Aggressive SMOTE-NC with verification
- SHAP PartitionExplainer with correlation clustering
- TabNet with balanced_accuracy metric
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier

# Advanced Libraries
try:
    import shap
    from imblearn.over_sampling import SMOTENC, SMOTE
except ImportError as e:
    print(f"[WARNING] Missing libraries: {e}. Some steps will be skipped.")
    shap = None
    SMOTENC = None
    SMOTE = None


class FinalProductionPipeline:
    def __init__(self, base_path, artifacts_dir="artifacts"):
        self.base_path = base_path
        self.artifacts_dir = artifacts_dir
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
            
        self.preprocessor = None
        self.feature_names = None
        self.label_encoder_binary = LabelEncoder()
        self.label_encoder_multi = LabelEncoder()
        self.categorical_indices = []
        
    def load_and_clean(self, fraction=0.3):
        print("\n" + "="*60)
        print("[STEP 1] Loading and Cleaning Data (Spark)")
        print("="*60)
        
        loader = DataLoader(self.base_path)
        df_spark = loader.load_data()
        
        if df_spark is None:
            sys.exit(1)
            
        print(f"  - Sampling {fraction*100:.0f}% of data...")
        df_sampled = df_spark.sample(withReplacement=False, fraction=fraction, seed=42)
        
        df = df_sampled.toPandas()
        print(f"  - Raw Shape: {df.shape}")
        
        # Aggressive Anti-Leakage
        print("  - [Anti-Leakage] Dropping Forbidden Columns...")
        df.columns = df.columns.str.strip()
        
        leakage_cols = ['Source IP', 'Destination IP', 'Timestamp', 'Flow ID', 
                        'Source Port', 'Destination Port', 'Unnamed: 0', 
                        'source_file', 'basename', 'SubType', 'Src IP', 'Dst IP', 
                        'MAC', 'Prob', 'Flow Duration', 'Protocol']
        
        existing_drop_cols = [c for c in leakage_cols if c in df.columns]
        df = df.drop(columns=existing_drop_cols)
        
        # Mode Imputation
        print("  - [Imputation] Filling NaNs with Mode...")
        for col in df.columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
        
        print(f"  - Cleaned Shape: {df.shape}")
        return df

    def preprocess(self, df):
        print("\n" + "="*60)
        print("[STEP 2] Preprocessing (One-Hot + Scaling)")
        print("="*60)
        
        X = df.drop(columns=['Label'])
        y_raw = df['Label']
        
        # Identify types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"  - Numeric: {len(numeric_features)}")
        print(f"  - Categorical: {len(categorical_features)}")
        
        # Safety Check for High Cardinality
        for col in list(categorical_features):
            unique_count = X[col].nunique()
            if unique_count > 50:
                print(f"  - [WARNING] Dropping high-cardinality: {col} ({unique_count})")
                X = X.drop(columns=[col])
                categorical_features.remove(col)
        
        # Label Encode categorical for SMOTE-NC
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        
        print("  - Label Encoding Categorical for SMOTE-NC...")
        self.cat_encoders = {}
        X_encoded = X.copy()
        for col in categorical_features:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            self.cat_encoders[col] = le
            
        self.categorical_indices = [X_encoded.columns.get_loc(c) for c in categorical_features]
        
        return X_encoded, y_raw

    def prepare_targets(self, y_raw):
        print("\n" + "="*60)
        print("[STEP 3] Preparing Dual-Track Targets")
        print("="*60)
        
        # Binary: Benign (0) vs Attack (1)
        y_binary = y_raw.apply(lambda x: 0 if x == 'Benign' else 1).values
        self.label_encoder_binary.fit(['Benign', 'Attack'])
        
        # Multiclass: Encode all unique labels
        y_multi = self.label_encoder_multi.fit_transform(y_raw)
        
        print(f"  - Binary Classes: [0=Benign, 1=Attack]")
        print(f"  - Multiclass Classes: {list(self.label_encoder_multi.classes_)}")
        
        # Save encoders
        joblib.dump(self.label_encoder_binary, os.path.join(self.artifacts_dir, "label_encoder_binary.pkl"))
        joblib.dump(self.label_encoder_multi, os.path.join(self.artifacts_dir, "label_encoder_multiclass.pkl"))
        
        return y_binary, y_multi

    def apply_smote_nc(self, X_train, y_train, target_name="Binary"):
        if not (SMOTENC or SMOTE):
            print("[WARNING] SMOTE not available. Skipping balancing.")
            return X_train, y_train
            
        print(f"\n  [SMOTE-NC for {target_name}]")
        print(f"  - Original Shape: {X_train.shape}")
        
        # Print class distribution BEFORE
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"  - Class Dist BEFORE: {dict(zip(unique, counts))}")
        
        # Apply SMOTE-NC with aggressive balancing
        if len(self.categorical_indices) > 0 and SMOTENC:
            smote = SMOTENC(categorical_features=self.categorical_indices, 
                           random_state=42, 
                           sampling_strategy='auto')  # 'auto' = match majority
            X_res, y_res = smote.fit_resample(X_train, y_train)
        elif SMOTE:
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X_res, y_res = smote.fit_resample(X_train, y_train)
        else:
            return X_train, y_train
            
        # Print class distribution AFTER
        unique_res, counts_res = np.unique(y_res, return_counts=True)
        print(f"  - Class Dist AFTER:  {dict(zip(unique_res, counts_res))}")
        print(f"  - Resampled Shape: {X_res.shape}")
        
        return X_res, y_res

    def finalize_features(self, df_train, df_val, df_test, feature_names):
        print("\n" + "="*60)
        print("[STEP 4] Finalizing Feature Engineering")
        print("="*60)
        
        # Define ColumnTransformer
        final_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
            ],
            verbose_feature_names_out=False
        )
        
        # Fit on Train
        X_train_final = final_preprocessor.fit_transform(df_train)
        X_val_final = final_preprocessor.transform(df_val)
        X_test_final = final_preprocessor.transform(df_test)
        
        # Get final feature names
        try:
            final_feature_names = final_preprocessor.get_feature_names_out().tolist()
        except:
            final_feature_names = [f"feat_{i}" for i in range(X_train_final.shape[1])]
            
        # Save
        self.preprocessor = final_preprocessor
        joblib.dump(final_preprocessor, os.path.join(self.artifacts_dir, "final_preprocessor.pkl"))
        joblib.dump(final_feature_names, os.path.join(self.artifacts_dir, "final_feature_names.pkl"))
        
        print(f"  - Final Feature Count: {len(final_feature_names)}")
        
        return X_train_final, X_val_final, X_test_final, final_feature_names

    def train_tabnet(self, X_train, y_train, X_val, y_val, name):
        print(f"\n" + "="*60)
        print(f"[STEP 5] Training TabNet: {name}")
        print("="*60)
        
        # Class Weights
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print(f"  - Class Weights: {class_weights}")
        
        clf = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            verbose=1
        )
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_name=['train', 'valid'],
            eval_metric=['balanced_accuracy'],  # Use balanced accuracy for imbalanced data
            max_epochs=50,
            patience=10,  # Increased patience
            batch_size=16384,
            virtual_batch_size=1024,
            num_workers=0,
            drop_last=False,
            weights=class_weights
        )
        
        # Save
        clf.save_model(os.path.join(self.artifacts_dir, name))
        print(f"  - Saved {name}.zip")
        
        return clf

    def evaluate(self, clf, X_test, y_test, name, target_names):
        print(f"\n[Evaluation] {name}")
        y_pred = clf.predict(X_test)
        
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        print(f"  >> Balanced Accuracy: {bal_acc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(os.path.join(self.artifacts_dir, f"cm_{name}.png"))
        plt.close()

    def run_shap(self, clf, X_train, X_test, feature_names, class_names):
        if not shap:
            print("[WARNING] SHAP not available. Skipping.")
            return

        print("\n" + "="*60)
        print("[STEP 6] SHAP PartitionExplainer (with Clustering)")
        print("="*60)
        
        try:
            # Sample background data
            X_summary = shap.utils.sample(X_test, 100)
            
            # Create masker with explicit clustering (FIX for .clustering error)
            masker = shap.maskers.Partition(X_summary, max_evals=2000, clustering="correlation")
            
            # Initialize PartitionExplainer
            explainer = shap.PartitionExplainer(clf.predict_proba, masker, output_names=class_names)
            
            # Explain a small sample
            print("  - Calculating SHAP values (15 samples)...")
            shap_values = explainer(X_test[:15])
            
            # Plot Beeswarm for Attack class (index 1)
            plt.figure()
            shap.plots.beeswarm(shap_values[:,:,1], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.artifacts_dir, "shap_partition_beeswarm.png"))
            plt.close()
            print("  - Saved shap_partition_beeswarm.png")
            
        except Exception as e:
            print(f"  - [SHAP Error] {e}")

    def run(self):
        # 1. Load
        df = self.load_and_clean(fraction=0.3)
        
        # 2. Preprocess (Label Encode for SMOTE)
        X_encoded, y_raw = self.preprocess(df)
        feature_names_encoded = X_encoded.columns.tolist()
        
        # 3. Prepare Targets
        y_binary, y_multi = self.prepare_targets(y_raw)
        
        # 4. Split (70/15/15) - Stratify on Multiclass
        X_train, X_temp, y_b_train, y_b_temp, y_m_train, y_m_temp = train_test_split(
            X_encoded.values, y_binary, y_multi, test_size=0.3, stratify=y_multi, random_state=42
        )
        X_val, X_test, y_b_val, y_b_test, y_m_val, y_m_test = train_test_split(
            X_temp, y_b_temp, y_m_temp, test_size=0.5, stratify=y_m_temp, random_state=42
        )
        
        # 5. SMOTE-NC for Binary
        print("\n" + "="*60)
        print("[STEP 3.5] Applying SMOTE-NC to Training Data")
        print("="*60)
        X_train_b_bal, y_train_b_bal = self.apply_smote_nc(X_train, y_b_train, "Binary")
        X_train_m_bal, y_train_m_bal = self.apply_smote_nc(X_train, y_m_train, "Multiclass")
        
        # 6. Finalize Features
        # Convert balanced arrays back to DataFrames for ColumnTransformer
        df_train_b = pd.DataFrame(X_train_b_bal, columns=feature_names_encoded)
        df_train_m = pd.DataFrame(X_train_m_bal, columns=feature_names_encoded)
        df_val = pd.DataFrame(X_val, columns=feature_names_encoded)
        df_test = pd.DataFrame(X_test, columns=feature_names_encoded)
        
        # Binary track
        X_train_b_final, X_val_final, X_test_final, final_feature_names = self.finalize_features(
            df_train_b, df_val, df_test, feature_names_encoded
        )
        
        # Multiclass track (reuse preprocessor)
        X_train_m_final = self.preprocessor.transform(df_train_m)
        
        # 7. Train Binary Model
        clf_binary = self.train_tabnet(X_train_b_final, y_train_b_bal, X_val_final, y_b_val, "tabnet_binary")
        self.evaluate(clf_binary, X_test_final, y_b_test, "Binary", ['Benign', 'Attack'])
        
        # 8. Train Multiclass Model
        clf_multi = self.train_tabnet(X_train_m_final, y_train_m_bal, X_val_final, y_m_val, "tabnet_multiclass")
        self.evaluate(clf_multi, X_test_final, y_m_test, "Multiclass", list(self.label_encoder_multi.classes_))
        
        # 9. SHAP (Binary)
        self.run_shap(clf_binary, X_train_b_final, X_test_final, final_feature_names, ['Benign', 'Attack'])
        
        print("\n" + "="*60)
        print("[DONE] Final Pipeline Complete.")
        print("="*60)


if __name__ == "__main__":
    pipeline = FinalProductionPipeline(base_path="data/raw/WiFi_and_MQTT")
    pipeline.run()
