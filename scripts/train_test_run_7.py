"""
train_test_run_7.py
========================
Mobile-First SOTA Transformer-TabNet Hybrid IoMT IDS
Target: High Accuracy, ~42ms Latency, ~0.026 MB Model
- One-Hot Encoding (All categorical)
- GA Feature Selection (15-18 features)
- Stratified Time-Split (All 6 classes)
- Tiny TabNet (~6.4K params)
- ReduceLROnPlateau Scheduler
- Soft SMOTE (20-30% of majority)
- Post-Training Quantization
- Stream Simulation & TabNet XAI
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
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# PyTorch
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

# Spark
from src.processing.loader import DataLoader
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import FloatType

# Sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# GA
try:
    from deap import base, creator, tools
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

# SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# =============================================================================
# TINY TABNET ATTENTION (~6.4K params with hidden_dim=32)
# =============================================================================
class TabNetAttention(nn.Module):
    """
    Lightweight TabNet-style attention for feature selection.
    ~6.4K params with hidden_dim=32.
    """
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        orig_shape = x.shape
        if len(orig_shape) == 3:
            b, s, f = x.shape
            x_flat = x.reshape(b * s, f)
        else:
            x_flat = x
        
        mask = torch.relu(self.fc1(x_flat))
        mask = self.bn1(mask)
        mask = torch.softmax(self.fc2(mask), dim=-1)
        out = x_flat * mask
        
        if len(orig_shape) == 3:
            out = out.reshape(b, s, f)
            mask = mask.reshape(b, s, f)
        return out, mask


# =============================================================================
# TRANSFORMER ENCODER (2 layers, 4 heads)
# =============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        
    def forward(self, x):
        return self.encoder(x)


# =============================================================================
# MOBILE HYBRID MODEL
# =============================================================================
class MobileHybridModel(nn.Module):
    """
    Transformer-TabNet Hybrid for Mobile Deployment.
    Target: ~0.026 MB after quantization.
    """
    NUM_CLASSES = 6
    
    def __init__(self, input_dim, seq_len=10, hidden_dim=32):
        super().__init__()
        self.seq_len = seq_len
        
        # TabNet: Feature Selection (~6.4K params)
        self.tabnet = TabNetAttention(input_dim, hidden_dim=hidden_dim)
        
        # Projection: input_dim -> 64 (for Transformer)
        self.projection = nn.Linear(input_dim, 64)
        
        # Transformer: Temporal (2 layers, 4 heads)
        self.transformer = TransformerBlock(d_model=64, nhead=4, num_layers=2)
        
        # Classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.NUM_CLASSES)
        )
        
        self.last_mask = None
        
    def forward(self, x):
        x_att, self.last_mask = self.tabnet(x)
        x_proj = self.projection(x_att)
        x_temp = self.transformer(x_proj)
        x_pool = x_temp.mean(dim=1)
        return self.classifier(x_pool)


# =============================================================================
# GENETIC ALGORITHM (15-18 features)
# =============================================================================
def ga_feature_selection(X_train, y_train, feature_names, target_features=17):
    if not DEAP_AVAILABLE:
        print("  [GA] DEAP unavailable. Using all features.")
        return list(range(len(feature_names))), feature_names
    
    print("\n" + "="*60)
    print(f"[GA] Selecting ~{target_features} Critical Features")
    print("="*60)
    
    n_features = X_train.shape[1]
    
    if hasattr(creator, 'FitnessMax'):
        del creator.FitnessMax
    if hasattr(creator, 'Individual'):
        del creator.Individual
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    prob_select = target_features / n_features
    toolbox.register("attr_bool", lambda: 1 if np.random.random() < prob_select else 0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    sample_size = min(5000, len(X_train))
    X_s, y_s = X_train[:sample_size], y_train[:sample_size]
    
    def evaluate(ind):
        sel = [i for i, b in enumerate(ind) if b == 1]
        if len(sel) == 0:
            return (0.0,)
        clf = DecisionTreeClassifier(max_depth=8, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X_s[:, sel], y_s, test_size=0.2, random_state=42)
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        # Penalty for deviating from target
        penalty = 0.02 * abs(len(sel) - target_features)
        return (acc - penalty,)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=30)
    
    for gen in range(20):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.7:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for m in offspring:
            if np.random.random() < 0.2:
                toolbox.mutate(m)
                del m.fitness.values
        for ind in [i for i in offspring if not i.fitness.valid]:
            ind.fitness.values = evaluate(ind)
        pop[:] = offspring
        if (gen + 1) % 5 == 0:
            best = max(i.fitness.values[0] for i in pop)
            n_sel = sum(tools.selBest(pop, 1)[0])
            print(f"  Gen {gen+1}: Best={best:.4f}, Features={n_sel}")
    
    best_ind = tools.selBest(pop, 1)[0]
    selected_idx = [i for i, b in enumerate(best_ind) if b == 1]
    selected_names = [feature_names[i] for i in selected_idx]
    
    print(f"  - Final: {len(selected_idx)} features")
    print(f"  - Selected: {selected_names[:10]}...")
    return selected_idx, selected_names


# =============================================================================
# WINDOWING (Majority Vote)
# =============================================================================
def create_windows(X, y, seq_len=10):
    n = len(X) // seq_len
    X_w = X[:n * seq_len].reshape(n, seq_len, -1)
    y_w = np.array([Counter(y[i*seq_len:(i+1)*seq_len]).most_common(1)[0][0] for i in range(n)])
    return X_w, y_w


# =============================================================================
# SOFT SMOTE (20-30% of majority)
# =============================================================================
def soft_smote(X, y, minority_ratio=0.25):
    if not SMOTE_AVAILABLE:
        return X, y
    
    n_classes = len(np.unique(y))
    if n_classes < 2:
        return X, y
    
    print("\n  [SOFT SMOTE] Upsampling to 25% of majority...")
    n_w, seq, feat = X.shape
    X_flat = X.reshape(n_w, -1)
    
    counts = Counter(y)
    print(f"    Before: {counts}")
    
    max_count = max(counts.values())
    target = int(max_count * minority_ratio)
    
    sampling_strategy = {}
    for cls, cnt in counts.items():
        if cnt < target:
            sampling_strategy[cls] = target
    
    if not sampling_strategy:
        print("    All classes >= 25% of majority. No SMOTE needed.")
        return X, y
    
    try:
        X_res, y_res = SMOTE(sampling_strategy=sampling_strategy, random_state=42).fit_resample(X_flat, y)
        print(f"    After: {Counter(y_res)}")
        return X_res.reshape(-1, seq, feat), y_res
    except Exception as e:
        print(f"    SMOTE failed: {e}")
        return X, y


# =============================================================================
# DATASET
# =============================================================================
class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


# =============================================================================
# MAIN PIPELINE
# =============================================================================
class MobileSOTAPipeline:
    ALL_CLASSES = ['Benign', 'DDoS', 'DoS', 'MQTT', 'Recon', 'Spoofing']
    
    def __init__(self, base_path, artifacts_dir="artifacts", seq_len=10):
        self.base_path = base_path
        self.artifacts_dir = artifacts_dir
        self.seq_len = seq_len
        os.makedirs(artifacts_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.ALL_CLASSES)  # Global 6 classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_data(self, fraction=0.1):
        print("\n" + "="*60)
        print("[STEP 1] Loading Data (Spark)")
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
        
        print(f"  - Samples: {len(df)}")
        print(f"  - Labels: {df['Label'].value_counts().to_dict()}")
        return df
    
    def stratified_time_split(self, df):
        """Split each class by time (index), then reassemble."""
        print("\n" + "="*60)
        print("[STEP 2] Stratified Time-Based Split")
        print("="*60)
        
        df = df.reset_index(drop=True)
        df['_idx'] = range(len(df))
        
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for label in df['Label'].unique():
            class_df = df[df['Label'] == label].copy()
            n = len(class_df)
            if n == 0:
                continue
            t1, t2 = int(n * 0.70), int(n * 0.85)
            train_dfs.append(class_df.iloc[:t1])
            val_dfs.append(class_df.iloc[t1:t2])
            test_dfs.append(class_df.iloc[t2:])
        
        df_train = pd.concat(train_dfs).sort_values('_idx').reset_index(drop=True)
        df_val = pd.concat(val_dfs).sort_values('_idx').reset_index(drop=True)
        df_test = pd.concat(test_dfs).sort_values('_idx').reset_index(drop=True)
        
        for d in [df_train, df_val, df_test]:
            d.drop(columns=['_idx'], inplace=True, errors='ignore')
        
        # Drop ID columns
        drop_cols = ['Timestamp', 'timestamp', 'Source IP', 'Destination IP', 
                     'Flow ID', 'Source Port', 'Destination Port', 'Unnamed: 0', 
                     'source_file', 'basename', 'SubType', 'Src IP', 'Dst IP', 
                     'MAC', 'Prob', 'Flow Duration', 'Time']
        for d in [df_train, df_val, df_test]:
            d.columns = d.columns.str.strip()
            d.drop(columns=[c for c in drop_cols if c in d.columns], inplace=True, errors='ignore')
        
        print(f"  - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
        for name, d in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
            print(f"    {name}: {d['Label'].value_counts().to_dict()}")
        
        return df_train, df_val, df_test
    
    def preprocess(self, df):
        """Mode imputation + One-Hot Encoding."""
        for c in df.columns:
            if df[c].isnull().any():
                mode = df[c].mode()
                df[c] = df[c].fillna(mode[0] if len(mode) > 0 else 0)
        
        X_raw = df.drop(columns=['Label'])
        y_raw = df['Label']
        
        # One-Hot Encode ALL categorical
        cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
        for c in list(cat_cols):
            if X_raw[c].nunique() > 50:
                X_raw = X_raw.drop(columns=[c])
                cat_cols.remove(c)
        if cat_cols:
            X = pd.get_dummies(X_raw, columns=cat_cols, dummy_na=True)
        else:
            X = X_raw
        X = X.astype(float)
        
        y = self.label_encoder.transform(y_raw)
        return X, y
    
    def align_features(self, X_train, X_val, X_test):
        train_cols = set(X_train.columns)
        for X in [X_val, X_test]:
            for col in train_cols - set(X.columns):
                X[col] = 0
            for col in set(X.columns) - train_cols:
                X.drop(columns=[col], inplace=True, errors='ignore')
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]
        return X_train, X_val, X_test
    
    def train(self, train_loader, val_loader, input_dim):
        print("\n" + "="*60)
        print("[STEP 5] Training (ReduceLROnPlateau)")
        print("="*60)
        
        model = MobileHybridModel(input_dim, self.seq_len, hidden_dim=32).to(self.device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  - Parameters: {n_params} ({n_params/1000:.1f}K)")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        
        best_acc = 0
        for epoch in range(40):
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()
            
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    preds = model(X_b.to(self.device)).argmax(dim=-1).cpu()
                    correct += (preds == y_b).sum().item()
                    total += len(y_b)
            
            acc = correct / total if total > 0 else 0
            scheduler.step(acc)
            
            if (epoch + 1) % 5 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}: Val Acc = {acc:.4f}, LR = {lr:.6f}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(self.artifacts_dir, "mobile_hybrid.pth"))
        
        model.load_state_dict(torch.load(os.path.join(self.artifacts_dir, "mobile_hybrid.pth")))
        print(f"  - Best Val Acc: {best_acc:.4f}")
        return model
    
    def quantize(self, model):
        print("\n[QUANTIZATION] (int8)")
        model.cpu().eval()
        quant = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        
        # Estimate size
        torch.save(quant.state_dict(), os.path.join(self.artifacts_dir, "mobile_hybrid_quant.pth"))
        size_mb = os.path.getsize(os.path.join(self.artifacts_dir, "mobile_hybrid_quant.pth")) / (1024 * 1024)
        print(f"  - Quantized Model Size: {size_mb:.3f} MB")
        return quant
    
    def stream_simulation(self, model, X_test):
        print("\n[STREAM SIMULATION] (Kappa Architecture)")
        model.eval().cpu()
        
        batch_size = 32
        n_batches = min(20, len(X_test) // batch_size)
        
        latencies = []
        for i in range(n_batches):
            batch = torch.FloatTensor(X_test[i*batch_size:(i+1)*batch_size])
            start = time.time()
            with torch.no_grad():
                _ = model(batch)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = np.mean(latencies)
        throughput = batch_size / (avg_latency / 1000)
        
        print(f"  - Avg Latency: {avg_latency:.2f} ms / batch of {batch_size}")
        print(f"  - Throughput: {throughput:.0f} samples/sec")
        return avg_latency
    
    def xai_tabnet_masks(self, model, X_test, feature_names):
        print("\n[XAI] TabNet Feature Masks")
        model.eval().cpu()
        
        with torch.no_grad():
            sample = torch.FloatTensor(X_test[:50])
            _ = model(sample)
            
            if model.last_mask is not None:
                mask = model.last_mask.mean(dim=(0, 1)).numpy()
                
                # Top 15 features
                top_idx = np.argsort(mask)[-15:][::-1]
                top_names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in top_idx]
                top_vals = mask[top_idx]
                
                print("  Top 10 Features:")
                for i, (n, v) in enumerate(zip(top_names[:10], top_vals[:10])):
                    print(f"    {i+1}. {n}: {v:.4f}")
                
                plt.figure(figsize=(10, 5))
                plt.bar(range(len(top_vals)), top_vals)
                plt.xticks(range(len(top_names)), top_names, rotation=45, ha='right')
                plt.title('TabNet Feature Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(self.artifacts_dir, "tabnet_masks.png"))
                plt.close()
                print("  - Saved tabnet_masks.png")
    
    def evaluate(self, model, test_loader):
        print("\n" + "="*60)
        print("[Evaluation]")
        print("="*60)
        
        model.eval().to(self.device)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                preds = model(X_b.to(self.device)).argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_b.numpy())
        
        report = classification_report(all_labels, all_preds, target_names=self.ALL_CLASSES, zero_division=0)
        print(report)
        
        with open(os.path.join(self.artifacts_dir, "classification_report.txt"), "w") as f:
            f.write(report)
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.ALL_CLASSES, yticklabels=self.ALL_CLASSES)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.artifacts_dir, "confusion_matrix.png"))
        plt.close()
        print("  - Saved confusion_matrix.png")
    
    def run(self):
        # 1. Load
        df = self.load_data(fraction=0.1)
        
        # 2. Stratified Time-Split
        df_train, df_val, df_test = self.stratified_time_split(df)
        
        # 3. Preprocess
        print("\n[STEP 3] Preprocessing (One-Hot Encoding)")
        X_train, y_train = self.preprocess(df_train)
        X_val, y_val = self.preprocess(df_val)
        X_test, y_test = self.preprocess(df_test)
        
        X_train, X_val, X_test = self.align_features(X_train, X_val, X_test)
        feature_names = X_train.columns.tolist()
        print(f"  - Features (One-Hot): {len(feature_names)}")
        
        X_train = X_train.values
        X_val = X_val.values
        X_test = X_test.values
        
        # 4. GA Feature Selection
        selected_idx, selected_names = ga_feature_selection(X_train, y_train, feature_names, target_features=17)
        X_train = X_train[:, selected_idx]
        X_val = X_val[:, selected_idx]
        X_test = X_test[:, selected_idx]
        
        # Scale
        print("\n[STEP 4] Scaling (StandardScaler)")
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        X_test_s = self.scaler.transform(X_test)
        
        # Windowing
        print("\n[WINDOWING]")
        X_tr_w, y_tr_w = create_windows(X_train_s, y_train, self.seq_len)
        X_val_w, y_val_w = create_windows(X_val_s, y_val, self.seq_len)
        X_te_w, y_te_w = create_windows(X_test_s, y_test, self.seq_len)
        print(f"  - Train: {X_tr_w.shape}, Val: {X_val_w.shape}, Test: {X_te_w.shape}")
        print(f"  - Train classes: {Counter(y_tr_w)}")
        
        # Soft SMOTE
        X_tr_w, y_tr_w = soft_smote(X_tr_w, y_tr_w, minority_ratio=0.25)
        
        # DataLoaders
        train_loader = TorchDataLoader(WindowDataset(X_tr_w, y_tr_w), batch_size=64, shuffle=True)
        val_loader = TorchDataLoader(WindowDataset(X_val_w, y_val_w), batch_size=64)
        test_loader = TorchDataLoader(WindowDataset(X_te_w, y_te_w), batch_size=64)
        
        # 5. Train
        input_dim = X_tr_w.shape[2]
        model = self.train(train_loader, val_loader, input_dim)
        
        # 6. Evaluate
        self.evaluate(model, test_loader)
        
        # 7. Quantize
        quant_model = self.quantize(model)
        
        # 8. Stream Simulation
        self.stream_simulation(quant_model, X_te_w)
        
        # 9. XAI
        self.xai_tabnet_masks(model, X_te_w, selected_names)
        
        # Save artifacts
        joblib.dump(self.scaler, os.path.join(self.artifacts_dir, "scaler.pkl"))
        joblib.dump(self.label_encoder, os.path.join(self.artifacts_dir, "label_encoder.pkl"))
        joblib.dump(selected_names, os.path.join(self.artifacts_dir, "selected_features.pkl"))
        
        print("\n" + "="*60)
        print("[DONE] Mobile SOTA Pipeline Complete!")
        print("="*60)


if __name__ == "__main__":
    MobileSOTAPipeline(base_path="data/raw/WiFi_and_MQTT", seq_len=10).run()
