import sys
import os
import time
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.loader import DataLoader
from pyspark.sql.functions import col, when, lit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier

# Advanced Libraries
try:
    import shap
    from alibi.explainers import AnchorTabular
    import dice_ml
    from imblearn.over_sampling import SMOTENC
except ImportError as e:
    print(f"[WARNING] Missing libraries: {e}. Some steps will be skipped.")
    shap = None
    AnchorTabular = None
    dice_ml = None
    SMOTENC = None

class FinalProductionPipeline:
    def __init__(self, base_path, artifacts_dir="artifacts"):
        self.base_path = base_path
        self.artifacts_dir = artifacts_dir
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
            
        self.preprocessor = None
        self.feature_names = None
        self.categorical_indices = [] # For SMOTE-NC
        
    def load_and_clean(self, fraction=0.3):
        print("\n[STEP 1] Loading and Cleaning Data (Spark)...")
        loader = DataLoader(self.base_path)
        df_spark = loader.load_data()
        
        if df_spark is None:
            sys.exit(1)
            
        # Sampling
        print(f"  - Sampling {fraction*100}% of data...")
        df_sampled = df_spark.sample(withReplacement=False, fraction=fraction, seed=42)
        
        # Convert to Pandas
        df = df_sampled.toPandas()
        
        # Aggressive Anti-Leakage & ID Dropping
        print("  - [Anti-Leakage] Dropping Forbidden Columns...")
        df.columns = df.columns.str.strip()
        
        leakage_cols = ['Source IP', 'Destination IP', 'Timestamp', 'Flow ID', 'Source Port', 'Destination Port', 'Unnamed: 0', 
                        'source_file', 'basename', 'SubType', 'Src IP', 'Dst IP', 'MAC', 'Prob', 'Flow Duration', 'Protocol']
        
        existing_drop_cols = [c for c in leakage_cols if c in df.columns]
        df = df.drop(columns=existing_drop_cols)
        
        # Mode Imputation (Pandas)
        print("  - [Imputation] Filling NaNs with Mode...")
        for col in df.columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
        
        return df

    def preprocess(self, df):
        print("\n[STEP 2] Preprocessing (One-Hot + Scaling)...")
        X = df.drop(columns=['Label'])
        y = df['Label']
        
        # Identify types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"  - Numeric: {len(numeric_features)}")
        print(f"  - Categorical: {len(categorical_features)}")
        
        # Safety Check for High Cardinality
        for col in categorical_features:
            unique_count = X[col].nunique()
            if unique_count > 50:
                print(f"  - [WARNING] Dropping high-cardinality column: {col} ({unique_count})")
                X = X.drop(columns=[col])
                categorical_features.remove(col)
        
        # Transform
        # We need to keep track of categorical indices for SMOTE-NC *after* transformation?
        # SMOTE-NC works on mixed data usually before OneHot, or we need to know which cols are cat.
        # Imblearn SMOTENC expects the dataset to be mixed (not fully encoded) or we specify indices.
        # BUT TabNet expects encoded data.
        # Strategy: 
        # 1. Split Train/Test
        # 2. Apply SMOTE-NC on Train (Mixed Data)
        # 3. Apply OneHot+Scaling on Train and Test
        
        # So we return X, y raw here, but we need to identify categorical indices for SMOTE-NC
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        
        # We need to convert categorical columns to strings to ensure SMOTE-NC handles them if they are object
        # Actually SMOTENC handles categorical features, but they need to be encoded as numbers usually or we pass indices.
        # Let's check SMOTENC docs mentally: "categorical_features : ndarray of shape (n_cat_features,) or (n_features,)"
        # It can handle strings if we use a pipeline? No, usually it expects numeric matrix where cat cols are label encoded.
        # Let's LabelEncode categorical features first for SMOTE-NC compatibility.
        
        print("  - Label Encoding Categorical Features for SMOTE-NC...")
        self.cat_encoders = {}
        X_encoded = X.copy()
        for col in categorical_features:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            self.cat_encoders[col] = le
            
        # Get indices of categorical columns
        self.categorical_indices = [X_encoded.columns.get_loc(c) for c in categorical_features]
        
        return X_encoded, y

    def apply_smote_nc(self, X_train, y_train):
        if not SMOTENC:
            print("[WARNING] SMOTE-NC not found. Skipping balancing.")
            return X_train, y_train
            
        print("\n[STEP 3] Balancing Classes with SMOTE-NC...")
        print(f"  - Original Shape: {X_train.shape}")
        print(f"  - Class Dist: {np.unique(y_train, return_counts=True)}")
        
        # SMOTE-NC
        # sampling_strategy='auto' resamples all classes to match majority
        # This might be too heavy if majority is huge.
        # Let's use 'auto' but be careful with memory. 30% data might be fine.
        
        if len(self.categorical_indices) > 0:
            smote = SMOTENC(categorical_features=self.categorical_indices, random_state=42, sampling_strategy='auto')
            X_res, y_res = smote.fit_resample(X_train, y_train)
        else:
            # Fallback to SMOTE if no cat features
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            
        print(f"  - Resampled Shape: {X_res.shape}")
        print(f"  - New Class Dist: {np.unique(y_res, return_counts=True)}")
        
        return X_res, y_res

    def finalize_preprocessing(self, X_train, X_val, X_test):
        print("\n[STEP 4] Finalizing Preprocessing (One-Hot + Scaling)...")
        
        # Now we have LabelEncoded categorical features. We should OneHot encode them for TabNet?
        # TabNet can handle embeddings if we pass cat_idxs.
        # But the user requested "One-Hot Encoding".
        # So we will OneHot encode the LabelEncoded columns.
        # And Scale the numeric ones.
        
        # We need to reconstruct DataFrames to use ColumnTransformer easily or just use arrays.
        # Let's use arrays and ColumnTransformer.
        
        # Problem: X_train is numpy array after SMOTE? Yes.
        # We need to know which columns are which.
        # X_encoded columns were: [numeric..., categorical...] (order preserved from df)
        # Let's assume order is preserved.
        
        # We need to separate numeric and categorical columns again based on indices.
        # This is getting complex with numpy arrays.
        # Alternative: Convert back to DataFrame?
        # Yes, let's convert X_train back to DF using feature names from X_encoded.
        
        # Wait, I didn't save feature names of X_encoded.
        # Let's fix that in preprocess.
        pass # Logic handled in run()

    def train_tabnet(self, X_train, y_train, X_val, y_val, feature_names):
        print("\n[STEP 5] Training TabNet (Transformer-Based)...")
        
        # TabNetClassifier
        clf = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            verbose=1
        )
        
        # Fit
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy'],
            max_epochs=50,
            patience=15,
            batch_size=16384,
            virtual_batch_size=1024,
            num_workers=0,
            drop_last=False
        )
        
        # Save
        clf.save_model(os.path.join(self.artifacts_dir, "tabnet_final"))
        print("  - Model Saved.")
        
        return clf

    def run_clinical_xai(self, clf, X_train, X_test, feature_names):
        if not (shap and AnchorTabular and dice_ml):
            return

        print("\n[STEP 6] Running Clinical XAI Suite (Binary)...")
        
        # 1. Partition SHAP (Transformer-Optimized)
        print("  - [SHAP] Partition Explainer...")
        # Partition explainer is good for correlated features and transformer models
        # It requires a masker.
        # For tabular, we use shap.maskers.Partition or Independent.
        # Let's use shap.maskers.Independent for simplicity and speed if Partition is complex to setup without hierarchy.
        # User asked for Partition SHAP.
        
        try:
            # We need a prediction function that outputs probabilities
            predict_fn = clf.predict_proba
            
            # Masker
            masker = shap.maskers.Independent(data=X_train[:100])
            
            # Explainer
            explainer = shap.PartitionExplainer(predict_fn, masker, feature_names=feature_names)
            
            # Explain
            shap_values = explainer(X_test[:15]) # Small sample
            
            # Plot (Beeswarm)
            # shap_values is an Explanation object
            # We want the values for the Attack class (index 1)
            plt.figure()
            shap.plots.beeswarm(shap_values[:,:,1], show=False)
            plt.savefig(os.path.join(self.artifacts_dir, "shap_partition.png"))
            plt.close()
        except Exception as e:
            print(f"  - [SHAP Error] {e}")

        # 2. Anchors
        print("  - [Anchors] Generating Rules...")
        predict_fn = lambda x: clf.predict(x)
        explainer_anchor = AnchorTabular(predict_fn, feature_names)
        explainer_anchor.fit(X_train)
        
        attacks = np.where(clf.predict(X_test) == 1)[0]
        if len(attacks) > 0:
            try:
                exp = explainer_anchor.explain(X_test[attacks[0]], threshold=0.95)
                print(f"    > Rule: {exp.anchor}")
                with open(os.path.join(self.artifacts_dir, "anchor_rule.txt"), "w") as f:
                    f.write(str(exp.anchor))
            except Exception as e:
                print(f"    > Anchor Error: {e}")

        # 3. DiCE
        print("  - [DiCE] Generating Counterfactuals...")
        class DiCEWrapper:
            def __init__(self, model):
                self.model = model
            def predict_proba(self, x):
                data = x.values if hasattr(x, "values") else x
                return self.model.predict_proba(data)
                
        dice_model = dice_ml.Model(model=DiCEWrapper(clf), backend="sklearn")
        d = dice_ml.Data(dataframe=pd.DataFrame(X_train, columns=feature_names).assign(Label=clf.predict(X_train)), 
                         continuous_features=feature_names, outcome_name='Label')
        exp_dice = dice_ml.Dice(d, dice_model, method="random")
        
        if len(attacks) > 0:
            q = pd.DataFrame([X_test[attacks[0]]], columns=feature_names)
            try:
                cf = exp_dice.generate_counterfactuals(q, total_CFs=1, desired_class=0)
                print("    > Counterfactual generated.")
                cf_json = cf.to_json()
                with open(os.path.join(self.artifacts_dir, "dice_cf.json"), "w") as f:
                    f.write(cf_json)
            except Exception as e:
                print(f"    > DiCE Error: {e}")

    def run(self):
        # 1. Load
        df = self.load_and_clean(fraction=0.3)
        
        # 2. Preprocess (Label Encode for SMOTE)
        X_encoded, y = self.preprocess(df)
        feature_names_encoded = X_encoded.columns.tolist()
        
        # 3. Targets (Binary for this pipeline as per XAI requirement, or Dual? User said "For the Binary classification output...")
        # User implies Binary focus for XAI. But "Dual-Track" was previous request. 
        # This request says "For the Binary classification output... implement...".
        # It doesn't explicitly say "Train ONLY Binary".
        # But step 3 says "Implement TabNetClassifier...". Singular.
        # Let's assume Binary for the main flow to support the Clinical XAI suite fully.
        y_binary = y.apply(lambda x: 0 if x == 'Benign' else 1).values
        
        # 4. Split (70/15/15)
        X_train, X_temp, y_train, y_temp = train_test_split(X_encoded.values, y_binary, test_size=0.3, stratify=y_binary, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        
        # 5. SMOTE-NC (Train only)
        X_train_bal, y_train_bal = self.apply_smote_nc(X_train, y_train)
        
        # 6. Final Preprocessing (One-Hot + Scaling)
        # We need to transform the LabelEncoded/Numeric mixed array into OneHot/Scaled array.
        # We need to know which columns are which.
        # self.categorical_indices tells us which columns in X_encoded are categorical.
        
        print("\n[STEP 4] Finalizing Feature Engineering...")
        
        # Reconstruct DFs for easy ColumnTransformer usage
        df_train = pd.DataFrame(X_train_bal, columns=feature_names_encoded)
        df_val = pd.DataFrame(X_val, columns=feature_names_encoded)
        df_test = pd.DataFrame(X_test, columns=feature_names_encoded)
        
        # Define Transformer
        # Numeric cols are those NOT in categorical_features
        # But feature_names_encoded includes all.
        # self.numeric_features and self.categorical_features hold original names.
        
        # We need to be careful: X_encoded has LabelEncoded values for cat features.
        # OneHotEncoder can take these integers if we tell it? Or we treat them as categorical.
        # OneHotEncoder expects 2D array.
        
        # Let's define the transformer on the columns
        # Numeric columns: Scale
        # Categorical columns (LabelEncoded): OneHot
        
        # Note: OneHotEncoder usually expects strings or integers. Integers are fine.
        
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
            
        # Save artifacts
        joblib.dump(final_preprocessor, os.path.join(self.artifacts_dir, "final_preprocessor.pkl"))
        joblib.dump(final_feature_names, os.path.join(self.artifacts_dir, "final_feature_names.pkl"))
        
        # 7. Train
        clf = self.train_tabnet(X_train_final, y_train_bal, X_val_final, y_val, final_feature_names)
        
        # 8. Evaluate
        print("\n[Evaluation]")
        y_pred = clf.predict(X_test_final)
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
        
        # 9. XAI
        self.run_clinical_xai(clf, X_train_final, X_test_final, final_feature_names)
        
        print("\n[DONE] Final Pipeline Complete.")

if __name__ == "__main__":
    pipeline = FinalProductionPipeline(base_path="data/raw/WiFi_and_MQTT")
    pipeline.run()
