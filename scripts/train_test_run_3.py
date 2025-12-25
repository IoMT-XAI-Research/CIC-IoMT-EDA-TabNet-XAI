import sys
import os
import time
import pandas as pd
import numpy as np
import joblib
import torch
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.loader import DataLoader
from pyspark.sql.functions import col, when, lit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier

# XAI Libraries (Try-Except for robustness)
try:
    import shap
    from alibi.explainers import AnchorTabular
    import dice_ml
except ImportError as e:
    print(f"[WARNING] XAI libraries not found: {e}. XAI steps will be skipped or limited.")
    shap = None
    AnchorTabular = None
    dice_ml = None

class IoMT_Pipeline:
    def __init__(self, base_path, artifacts_dir="artifacts"):
        self.base_path = base_path
        self.artifacts_dir = artifacts_dir
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
            
        self.scaler = None
        self.preprocessor = None # For OneHot + Scaling
        self.label_encoder_binary = None
        self.label_encoder_multi = None
        
    def load_and_clean_data(self, fraction=0.7):
        """
        Loads data using PySpark, samples it, and converts to Pandas.
        """
        print(f"\n[STEP 1] Loading Data (Fraction: {fraction})...")
        loader = DataLoader(self.base_path)
        
        # Load Train and Test
        df_train_spark = loader.load_data() # This loads all CSVs recursively
        # Note: The original loader loads everything. We can filter if needed, 
        # but here we assume we load everything and then split.
        # Actually, the previous script loaded train and test separately. 
        # Let's stick to the robust pattern of loading everything found in the path.
        
        if df_train_spark is None:
            print("[ERROR] No data found.")
            sys.exit(1)
            
        # Preprocess in Spark (Inf -> NaN -> Median) - Loader does this.
        # But user wants "Fill NaNs with 0".
        # We can override or just do it in Pandas.
        # Let's do it in Pandas for strict compliance with "Handle NaNs by filling with 0".
        
        # Sample
        print(f"  - Sampling {fraction*100}% of data...")
        df_sampled = df_train_spark.sample(withReplacement=False, fraction=fraction, seed=42)
        
        print("  - Converting to Pandas...")
        df = df_sampled.toPandas()
        
        # Clean Metadata
        print("  - Cleaning Metadata...")
        # Strip whitespace from column names to ensure we match ' Flow ID' or 'Flow ID'
        df.columns = df.columns.str.strip()
        
        cols_to_drop = ['source_file', 'basename', 'SubType', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'MAC', 'Prob']
        # Also drop any columns that look like IDs if they exist
        
        existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
        if existing_drop_cols:
            print(f"  - Dropping columns: {existing_drop_cols}")
            df = df.drop(columns=existing_drop_cols)
        
        # Handle NaNs (User Requirement: Fill with 0)
        print("  - Filling NaNs with 0...")
        df = df.fillna(0)
        
        return df

    def preprocess_features(self, df):
        """
        Applies One-Hot Encoding and StandardScaler.
        """
        print("\n[STEP 2] Preprocessing Features (One-Hot + Scaling)...")
        
        X = df.drop(columns=['Label'])
        y = df['Label']
        
        # Identify columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"  - Numeric Features ({len(numeric_features)}): {numeric_features}")
        print(f"  - Categorical Features ({len(categorical_features)}): {categorical_features}")
        
        # Safety Check for High Cardinality
        for col in categorical_features:
            unique_count = X[col].nunique()
            print(f"    > {col}: {unique_count} unique values")
            if unique_count > 50:
                print(f"    [WARNING] High cardinality detected in '{col}'. Dropping it to prevent memory explosion.")
                X = X.drop(columns=[col])
                categorical_features.remove(col)
        
        # Define Transformer
        # OneHotEncoder: handle_unknown='ignore' is crucial for production
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            verbose_feature_names_out=False
        )
        
        # Fit and Transform
        print("  - Fitting Preprocessor...")
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get Feature Names
        # This might fail in older sklearn versions, but we assume recent.
        try:
            feature_names = self.preprocessor.get_feature_names_out()
        except:
            # Fallback
            feature_names = numeric_features + list(self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
            
        # Save Preprocessor
        joblib.dump(self.preprocessor, os.path.join(self.artifacts_dir, "preprocessor.pkl"))
        joblib.dump(feature_names, os.path.join(self.artifacts_dir, "feature_names_v3.pkl"))
        
        return X_processed, y, feature_names

    def prepare_targets(self, y):
        """
        Prepares Binary and Multiclass targets.
        """
        print("\n[STEP 3] Preparing Targets...")
        
        # Binary: Benign (0) vs Attack (1)
        y_binary = y.apply(lambda x: 0 if x == 'Benign' else 1).values
        
        # Multiclass: Encode labels
        # Ensure specific classes: Benign, DoS, DDoS, MQTT, Recon, Spoofing
        # Map anything else to 'Other' or handle? Assuming dataset is clean/filtered.
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder_multi = LabelEncoder()
        y_multi = self.label_encoder_multi.fit_transform(y)
        
        # Save Encoder
        joblib.dump(self.label_encoder_multi, os.path.join(self.artifacts_dir, "label_encoder_multi.pkl"))
        
        print(f"  - Binary Classes: {np.unique(y_binary)}")
        print(f"  - Multiclass Classes: {self.label_encoder_multi.classes_}")
        
        return y_binary, y_multi

    def split_data(self, X, y_binary, y_multi):
        """
        Stratified Split: 70% Train, 15% Val, 15% Test.
        """
        print("\n[STEP 4] Splitting Data (70/15/15)...")
        
        # Split into Train (70%) and Temp (30%)
        # Stratify based on Multiclass to ensure all classes are represented
        X_train, X_temp, y_b_train, y_b_temp, y_m_train, y_m_temp = train_test_split(
            X, y_binary, y_multi, test_size=0.3, stratify=y_multi, random_state=42
        )
        
        # Split Temp into Val (50% of 30% = 15%) and Test (50% of 30% = 15%)
        X_val, X_test, y_b_val, y_b_test, y_m_val, y_m_test = train_test_split(
            X_temp, y_b_temp, y_m_temp, test_size=0.5, stratify=y_m_temp, random_state=42
        )
        
        print(f"  - Train: {X_train.shape}")
        print(f"  - Val:   {X_val.shape}")
        print(f"  - Test:  {X_test.shape}")
        
        return (X_train, X_val, X_test), (y_b_train, y_b_val, y_b_test), (y_m_train, y_m_val, y_m_test)

    def train_tabnet(self, X_train, y_train, X_val, y_val, name="model"):
        """
        Trains a TabNet model.
        """
        print(f"\n[STEP 5] Training {name}...")
        
        # Class Weights
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
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
            eval_metric=['accuracy'],
            max_epochs=50,
            patience=15,
            batch_size=16384,
            virtual_batch_size=1024,
            num_workers=0,
            drop_last=False,
            weights=class_weights
        )
        
        # Save
        save_path = os.path.join(self.artifacts_dir, f"tabnet_{name}")
        clf.save_model(save_path)
        print(f"  - Saved {name} to {save_path}.zip")
        
        return clf

    def generate_xai(self, clf, X_train, X_test, feature_names):
        """
        Generates Anchor, SHAP, and Counterfactual explanations for the Binary Model.
        """
        print("\n[STEP 6] Generating XAI Explanations (Binary Model)...")
        
        if not (shap and AnchorTabular and dice_ml):
            print("[WARNING] Skipping XAI due to missing libraries.")
            return

        # 1. SHAP
        print("  - [SHAP] Calculating Feature Importance...")
        # Use small background for speed
        background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        
        # Explain 10 samples
        X_shap = X_test[:10]
        shap_values = explainer.shap_values(X_shap)
        
        # Plot
        plt.figure()
        shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
        plt.savefig(os.path.join(self.artifacts_dir, "shap_summary_v3.png"))
        plt.close()
        
        # 2. Anchors (Alibi)
        print("  - [Anchors] Generating Rules...")
        predict_fn = lambda x: clf.predict(x)
        explainer_anchor = AnchorTabular(predict_fn, feature_names)
        explainer_anchor.fit(X_train)
        
        # Explain an attack instance
        # Find an attack in X_test
        attack_indices = np.where(clf.predict(X_test) == 1)[0]
        if len(attack_indices) > 0:
            idx = attack_indices[0]
            explanation = explainer_anchor.explain(X_test[idx], threshold=0.95)
            print(f"    > Anchor Rule for Instance {idx}: {explanation.anchor}")
            with open(os.path.join(self.artifacts_dir, "anchor_rule.txt"), "w") as f:
                f.write(str(explanation.anchor))
        
        # 3. DiCE (Counterfactuals)
        print("  - [DiCE] Generating Counterfactuals...")
        # DiCE requires a dataframe
        d = dice_ml.Data(dataframe=pd.DataFrame(X_train, columns=feature_names).assign(Label=clf.predict(X_train)), 
                         continuous_features=feature_names, 
                         outcome_name='Label')
        m = dice_ml.Model(model=clf, backend="sklearn")
        exp = dice_ml.Dice(d, m, method="random")
        
        if len(attack_indices) > 0:
            idx = attack_indices[0]
            # Generate CF to flip Attack (1) to Benign (0)
            query_instance = pd.DataFrame([X_test[idx]], columns=feature_names)
            dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=1, desired_class=0)
            
            print("    > Counterfactual generated.")
            # Save as text or json
            cf_json = dice_exp.to_json()
            with open(os.path.join(self.artifacts_dir, "dice_counterfactual.json"), "w") as f:
                f.write(cf_json)

    def run(self):
        # Load
        df = self.load_and_clean_data(fraction=0.7)
        
        # Preprocess
        X, y, feature_names = self.preprocess_features(df)
        
        # Targets
        y_binary, y_multi = self.prepare_targets(y)
        
        # Split
        (X_train, X_val, X_test), (y_b_train, y_b_val, y_b_test), (y_m_train, y_m_val, y_m_test) = \
            self.split_data(X, y_binary, y_multi)
            
        # Train Binary
        clf_binary = self.train_tabnet(X_train, y_b_train, X_val, y_b_val, name="binary_model")
        
        # Evaluate Binary
        y_b_pred = clf_binary.predict(X_test)
        print("\n[Binary Model Report]")
        print(classification_report(y_b_test, y_b_pred))
        
        # Train Multiclass
        clf_multi = self.train_tabnet(X_train, y_m_train, X_val, y_m_val, name="multi_model")
        
        # Evaluate Multiclass
        y_m_pred = clf_multi.predict(X_test)
        print("\n[Multiclass Model Report]")
        print(classification_report(y_m_test, y_m_pred, target_names=self.label_encoder_multi.classes_))
        
        # XAI (Binary Only)
        self.generate_xai(clf_binary, X_train, X_test, feature_names)
        
        print("\n[DONE] Pipeline Complete.")

if __name__ == "__main__":
    pipeline = IoMT_Pipeline(base_path="data/raw/WiFi_and_MQTT")
    pipeline.run()
