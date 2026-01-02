"""
train_test_run_6.py
========================
Rule-Based IDS Pipeline (Gaikwad et al., 2014 Replication) - FIXED
- Phase 1: Genetic Algorithm Feature Selection (with SMOTE-NC)
- Phase 2: Ripple Down Rule (RDR) Classification
- NO SCALING for interpretable rules
- XAI: Inherent Interpretability + Anchor/DiCE
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.loader import DataLoader
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import FloatType
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text

# Genetic Algorithm
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    print("[WARNING] DEAP not installed. GA will be skipped.")
    DEAP_AVAILABLE = False

# SMOTE
try:
    from imblearn.over_sampling import SMOTENC, SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("[WARNING] imbalanced-learn not installed. SMOTE will be skipped.")
    SMOTE_AVAILABLE = False

# XAI Libraries
try:
    from alibi.explainers import AnchorTabular
    import dice_ml
except ImportError:
    print("[WARNING] XAI libraries not available.")
    AnchorTabular = None
    dice_ml = None


# =============================================================================
# CUSTOM RIPPLE DOWN RULE LEARNER (with predict_proba for DiCE)
# =============================================================================
class RippleDownRuleLearner:
    """
    Simplified Ripple Down Rule Learner with probability support.
    Uses Decision Tree as the "Exception Engine".
    """
    def __init__(self, max_depth=5, min_samples_split=50):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.default_class = None
        self.rules = []
        self.n_classes_ = 2
        
    def fit(self, X, y, feature_names=None):
        # Default Rule: Majority Class
        self.default_class = Counter(y).most_common(1)[0][0]
        self.n_classes_ = len(np.unique(y))
        
        # Use DecisionTree as the "Exception Engine"
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )
        self.tree.fit(X, y)
        
        # Extract rules
        if feature_names is not None:
            self.rules = self._extract_rules(feature_names)
            
    def predict(self, X):
        return self.tree.predict(X)
    
    def predict_proba(self, X):
        """
        Returns probability estimates.
        Uses the underlying tree's predict_proba if available.
        """
        return self.tree.predict_proba(X)
    
    def _extract_rules(self, feature_names, max_rules=None):
        """Extract ALL rules from the Decision Tree."""
        tree_rules = export_text(self.tree, feature_names=feature_names)
        lines = tree_rules.split('\n')
        rules = []
        current_rule = []
        for line in lines:
            if 'class:' in line:
                rules.append(' AND '.join(current_rule) + ' => ' + line.strip())
                current_rule = []
            elif '|---' in line:
                cond = line.replace('|---', '').strip()
                current_rule.append(cond)
        return rules if max_rules is None else rules[:max_rules]
    
    def get_rules(self, max_rules=None):
        """Get rules. If max_rules is None, return all rules."""
        return self.rules if max_rules is None else self.rules[:max_rules]
    
    def get_full_tree_text(self, feature_names):
        """Get the full decision tree as text for auditing."""
        return export_text(self.tree, feature_names=feature_names)


# =============================================================================
# GENETIC ALGORITHM FEATURE SELECTION
# =============================================================================
def ga_feature_selection(X_train, y_train, feature_names, 
                         population_size=20, n_generations=20, 
                         crossover_prob=0.7, mutation_prob=0.033):
    """
    Genetic Algorithm for Feature Selection.
    Paper Parameters: Pop=20, Gen=20, Crossover=0.7, Mutation=0.033
    """
    if not DEAP_AVAILABLE:
        print("[WARNING] DEAP not available. Returning all features.")
        return list(range(len(feature_names))), feature_names
    
    print("\n" + "="*60)
    print("[PHASE 1] Genetic Algorithm Feature Selection")
    print("="*60)
    print(f"  - Population Size: {population_size}")
    print(f"  - Generations: {n_generations}")
    print(f"  - Crossover Prob: {crossover_prob}")
    print(f"  - Mutation Prob: {mutation_prob}")
    
    n_features = X_train.shape[1]
    
    # Clear any previous DEAP setup
    if hasattr(creator, 'FitnessMax'):
        del creator.FitnessMax
    if hasattr(creator, 'Individual'):
        del creator.Individual
    
    # DEAP Setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Fitness Function
    def evaluate(individual):
        selected_idx = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected_idx) == 0:
            return (0.0,)
        
        X_sel = X_train[:, selected_idx]
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_train, test_size=0.2, random_state=42)
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        return (acc,)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_prob)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run GA
    print("  - Running GA...")
    pop = toolbox.population(n=population_size)
    
    for gen in range(n_generations):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if np.random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop[:] = offspring
        
        fits = [ind.fitness.values[0] for ind in pop]
        if (gen + 1) % 5 == 0:
            print(f"    Gen {gen+1}: Best Fitness = {max(fits):.4f}")
    
    # Get best individual
    best_ind = tools.selBest(pop, 1)[0]
    selected_idx = [i for i, bit in enumerate(best_ind) if bit == 1]
    selected_features = [feature_names[i] for i in selected_idx]
    
    print(f"  - Selected {len(selected_idx)}/{n_features} features")
    
    return selected_idx, selected_features


# =============================================================================
# MAIN PIPELINE
# =============================================================================
class GA_RDR_Pipeline:
    def __init__(self, base_path, artifacts_dir="artifacts"):
        self.base_path = base_path
        self.artifacts_dir = artifacts_dir
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
            
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.categorical_indices = []
        self.categorical_features = []
        self.numeric_features = []
        
    def load_and_clean(self, fraction=0.3):
        print("\n" + "="*60)
        print("[STEP 1] Loading and Cleaning Data (Spark)")
        print("="*60)
        
        loader = DataLoader(self.base_path)
        df_spark = loader.load_data()
        
        if df_spark is None:
            sys.exit(1)
        
        # =====================================================
        # FIX 1: Cast Rate and Variance to FloatType IMMEDIATELY
        # =====================================================
        print("  - [FIX] Casting Rate/Variance to FloatType...")
        if 'Rate' in df_spark.columns:
            df_spark = df_spark.withColumn("Rate", col("Rate").cast(FloatType()))
        if 'Variance' in df_spark.columns:
            df_spark = df_spark.withColumn("Variance", col("Variance").cast(FloatType()))
        
        # =====================================================
        # FIX NEW: Sanitize Infinity values -> Replace with None
        # =====================================================
        print("  - [FIX] Sanitizing Infinity values (replacing with None)...")
        numeric_cols = [f.name for f in df_spark.schema.fields 
                       if str(f.dataType) in ['FloatType()', 'DoubleType()', 'IntegerType()', 'LongType()']]
        
        for c_name in numeric_cols:
            df_spark = df_spark.withColumn(c_name, 
                when(col(c_name) == float("inf"), lit(None))
                .when(col(c_name) == float("-inf"), lit(None))
                .otherwise(col(c_name))
            )
            
        print(f"  - Sampling {fraction*100:.0f}% of data...")
        df_sampled = df_spark.sample(withReplacement=False, fraction=fraction, seed=42)
        df = df_sampled.toPandas()
        
        # =====================================================
        # FIX 2: Remap Empty Labels to 'MQTT'
        # =====================================================
        print("  - [FIX] Remapping empty Labels to 'MQTT'...")
        df['Label'] = df['Label'].replace('', 'MQTT')
        df['Label'] = df['Label'].fillna('MQTT')
        
        print(f"  - Label Distribution: {df['Label'].value_counts().to_dict()}")
        
        # Anti-Leakage
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
        for col_name in df.columns:
            if df[col_name].isnull().any():
                mode_val = df[col_name].mode()[0]
                df[col_name] = df[col_name].fillna(mode_val)
        
        print(f"  - Cleaned Shape: {df.shape}")
        return df

    def preprocess(self, df):
        print("\n" + "="*60)
        print("[STEP 2] Preprocessing (One-Hot ONLY - NO SCALING)")
        print("="*60)
        
        X = df.drop(columns=['Label'])
        y_raw = df['Label']
        
        # Encode target
        y = self.label_encoder.fit_transform(y_raw)
        joblib.dump(self.label_encoder, os.path.join(self.artifacts_dir, "label_encoder_rdr.pkl"))
        
        # Identify types
        self.numeric_features = X.select_dtypes(include=['int64', 'float64', 'float32']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"  - Numeric: {len(self.numeric_features)}")
        print(f"  - Categorical: {len(self.categorical_features)}")
        
        # Safety Check
        for col_name in list(self.categorical_features):
            if X[col_name].nunique() > 50:
                print(f"  - [WARNING] Dropping high-cardinality: {col_name}")
                X = X.drop(columns=[col_name])
                self.categorical_features.remove(col_name)
        
        # =====================================================
        # FIX 3: NO StandardScaler - Use Raw Data for Interpretability
        # Only OneHotEncode categorical features
        # =====================================================
        print("  - [NO SCALING] Using raw numeric values for interpretable rules...")
        
        if len(self.categorical_features) > 0:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', self.numeric_features),  # NO SCALING
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
                ],
                verbose_feature_names_out=False
            )
            X_processed = self.preprocessor.fit_transform(X)
            
            try:
                feature_names = self.preprocessor.get_feature_names_out().tolist()
            except:
                feature_names = self.numeric_features + [f"cat_{i}" for i in range(X_processed.shape[1] - len(self.numeric_features))]
        else:
            # No categorical features - just use numeric
            X_processed = X[self.numeric_features].values
            feature_names = self.numeric_features
            self.preprocessor = None
            
        # Get categorical indices for SMOTE-NC (after one-hot, these are at the end)
        self.categorical_indices = list(range(len(self.numeric_features), X_processed.shape[1]))
        
        joblib.dump(self.preprocessor, os.path.join(self.artifacts_dir, "preprocessor_rdr.pkl"))
        joblib.dump(feature_names, os.path.join(self.artifacts_dir, "feature_names_rdr.pkl"))
        
        print(f"  - Final Feature Count: {len(feature_names)}")
        return X_processed, y, feature_names

    def apply_smote_nc(self, X_train, y_train):
        if not SMOTE_AVAILABLE:
            print("[WARNING] SMOTE not available. Skipping balancing.")
            return X_train, y_train
            
        print("\n" + "="*60)
        print("[STEP 3] Balancing with SMOTE-NC")
        print("="*60)
        
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"  - Class Dist BEFORE: {dict(zip(unique, counts))}")
        
        if len(self.categorical_indices) > 0:
            smote = SMOTENC(categorical_features=self.categorical_indices, 
                           random_state=42, sampling_strategy='auto')
            X_res, y_res = smote.fit_resample(X_train, y_train)
        else:
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X_res, y_res = smote.fit_resample(X_train, y_train)
            
        unique_res, counts_res = np.unique(y_res, return_counts=True)
        print(f"  - Class Dist AFTER:  {dict(zip(unique_res, counts_res))}")
        print(f"  - Resampled Shape: {X_res.shape}")
        
        return X_res, y_res

    def run_xai(self, model, X_train, X_test, feature_names):
        print("\n" + "="*60)
        print("[STEP 6] XAI: Anchors & DiCE")
        print("="*60)
        
        # 1. Inherent Interpretability: RDR Rules (ALL RULES)
        print("  - [RDR Rules] Extracting ALL learned rules...")
        rules = model.get_rules()  # Now returns ALL rules
        print(f"  - Total Rules Extracted: {len(rules)}")
        print(f"  - Top 5 Rules:")
        for i, rule in enumerate(rules[:5], 1):
            print(f"    {i}. {rule}")
        
        # Save ALL rules to file
        with open(os.path.join(self.artifacts_dir, "rdr_rules.txt"), "w") as f:
            f.write(f"=== RDR Rules ({len(rules)} total) ===\n\n")
            for i, rule in enumerate(rules, 1):
                f.write(f"{i}. {rule}\n")
        
        # Save full decision tree text for auditing
        full_tree = model.get_full_tree_text(feature_names)
        with open(os.path.join(self.artifacts_dir, "rdr_full_tree.txt"), "w") as f:
            f.write("=== Full Decision Tree Structure ===\n\n")
            f.write(full_tree)
        print(f"  - Saved rdr_rules.txt and rdr_full_tree.txt")
        
        # 2. Anchors
        if AnchorTabular:
            print("  - [Anchors] Generating rules...")
            try:
                predict_fn = lambda x: model.predict(x)
                explainer = AnchorTabular(predict_fn, feature_names)
                explainer.fit(X_train[:1000])
                
                attacks = np.where(model.predict(X_test) == 1)[0]
                if len(attacks) > 0:
                    exp = explainer.explain(X_test[attacks[0]], threshold=0.85)
                    print(f"    > Anchor: {exp.anchor}")
            except Exception as e:
                print(f"    > Anchor Error: {e}")
        
        # 3. DiCE (with predict_proba support)
        if dice_ml:
            print("  - [DiCE] Generating counterfactuals...")
            try:
                dice_model = dice_ml.Model(model=model, backend="sklearn")
                d = dice_ml.Data(
                    dataframe=pd.DataFrame(X_train, columns=feature_names).assign(Label=model.predict(X_train)),
                    continuous_features=feature_names, 
                    outcome_name='Label'
                )
                exp_dice = dice_ml.Dice(d, dice_model, method="random")
                
                attacks = np.where(model.predict(X_test) == 1)[0]
                if len(attacks) > 0:
                    q = pd.DataFrame([X_test[attacks[0]]], columns=feature_names)
                    cf = exp_dice.generate_counterfactuals(q, total_CFs=1, desired_class=0)
                    print("    > Counterfactual generated.")
                    cf_json = cf.to_json()
                    with open(os.path.join(self.artifacts_dir, "dice_cf_rdr.json"), "w") as f:
                        f.write(cf_json)
            except Exception as e:
                print(f"    > DiCE Error: {e}")

    def run(self):
        start_time = time.time()
        
        # 1. Load
        df = self.load_and_clean(fraction=0.1)
        
        # 2. Preprocess (NO SCALING)
        X, y, feature_names = self.preprocess(df)
        
        # 3. Split (70/15/15)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        
        # 4. SMOTE-NC (BEFORE GA)
        X_train_bal, y_train_bal = self.apply_smote_nc(X_train, y_train)
        
        # 5. GA Feature Selection (Paper Parameters)
        selected_idx, selected_features = ga_feature_selection(
            X_train_bal, y_train_bal, feature_names,
            population_size=20,
            n_generations=20,
            crossover_prob=0.7,
            mutation_prob=0.033
        )
        
        # Apply selection
        X_train_sel = X_train_bal[:, selected_idx]
        X_val_sel = X_val[:, selected_idx]
        X_test_sel = X_test[:, selected_idx]
        
        joblib.dump(selected_features, os.path.join(self.artifacts_dir, "ga_selected_features.pkl"))
        
        # 6. RDR Training
        print("\n" + "="*60)
        print("[PHASE 2] Ripple Down Rule Learner")
        print("="*60)
        
        train_start = time.time()
        rdr = RippleDownRuleLearner(max_depth=5, min_samples_split=50)
        rdr.fit(X_train_sel, y_train_bal, feature_names=selected_features)
        train_time = time.time() - train_start
        print(f"  - Model Building Time: {train_time:.2f}s")
        
        joblib.dump(rdr, os.path.join(self.artifacts_dir, "rdr_model.pkl"))
        
        # 7. Evaluation
        print("\n" + "="*60)
        print("[STEP 5] Evaluation")
        print("="*60)
        
        y_pred = rdr.predict(X_test_sel)
        acc = accuracy_score(y_test, y_pred)
        
        # FPR Calculation
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            fpr = 0  # Multiclass - skip simple FPR
        
        print(f"  - Accuracy: {acc:.4f}")
        print(f"  - False Positive Rate (FPR): {fpr:.4f}")
        print(f"  - Model Building Time: {train_time:.2f}s")
        
        print("\nClassification Report:")
        target_names = [str(c) for c in self.label_encoder.classes_]
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix - RDR')
        plt.savefig(os.path.join(self.artifacts_dir, "cm_rdr.png"))
        plt.close()
        
        # 8. XAI
        self.run_xai(rdr, X_train_sel, X_test_sel, selected_features)
        
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"[DONE] GA + RDR Pipeline Complete. Total Time: {total_time:.2f}s")
        print("="*60)


if __name__ == "__main__":
    pipeline = GA_RDR_Pipeline(base_path="data/raw/WiFi_and_MQTT")
    pipeline.run()
