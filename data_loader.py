import pandas as pd
import numpy as np
import torch
import os
import glob
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# NSL-KDD column names (41 features + label + difficulty = 43 total)
COL_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']


# =============================================================================
# Pipeline A: CNN-LSTM Preprocessing (LabelEncoded, 41 features)
# =============================================================================
def load_and_preprocess_cnn_lstm(filepath, label_encoders=None, scaler=None):
    """Preprocessing for CNN-LSTM model.
    
    Uses LabelEncoding for categoricals (preserves spatial locality for Conv1d).
    All 41 features kept (including num_outbound_cmds).    StandardScaler on all features.
    
    Args:
        filepath: Path to NSL-KDD data file.
        label_encoders: Dict of pre-fitted LabelEncoders from training data. 
                        If None, fits new encoders (use for training set).
        scaler: Pre-fitted StandardScaler. If None, fits new one (use for training set).
    
    Returns:
        X, y, label_encoders, scaler
    """
    df = pd.read_csv(filepath, names=COL_NAMES)
    df = df.drop('difficulty', axis=1)
    
    # Binary label: normal=0, attack=1
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # LabelEncode categoricals — fitted on train, applied to both
    fit_new = label_encoders is None
    if fit_new:
        label_encoders = {}
    
    for col in CATEGORICAL_COLS:
        if fit_new:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            # Handle unseen categories in test set — map to a default
            df[col] = df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.longlong)
    
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, label_encoders, scaler


# =============================================================================
# Pipeline B: MLP Preprocessing (One-Hot Encoded, ~122 features)
# =============================================================================
def load_and_preprocess_mlp(filepath, train_columns=None, scaler=None):
    """Preprocessing for MLP model.
    
    Uses One-Hot Encoding for categoricals (no ordinal assumption).
    Drops num_outbound_cmds (all zeros — zero information).
    StandardScaler on numerical columns only (one-hot stays 0/1).
    
    Args:
        filepath: Path to NSL-KDD data file.
        train_columns: List of column names from training data for alignment.
                       If None, stores columns (use for training set).
        scaler: Pre-fitted StandardScaler for numerical columns.
                If None, fits new one (use for training set).
    
    Returns:
        X, y, train_columns, scaler, numerical_cols
    """
    df = pd.read_csv(filepath, names=COL_NAMES)
    df = df.drop('difficulty', axis=1)
    
    # Binary label
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    y = df['label'].values.astype(np.longlong)
    df = df.drop('label', axis=1)
    
    # Drop zero-variance feature
    df = df.drop('num_outbound_cmds', axis=1)
    
    # Identify numerical columns BEFORE one-hot encoding
    numerical_cols = [c for c in df.columns if c not in CATEGORICAL_COLS]
    
    # One-Hot Encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, dtype=np.float32)
    
    # Align columns between train and test
    if train_columns is None:
        # Training set — store column order
        train_columns = df.columns.tolist()
    else:
        # Test set — reindex to match training columns (fill missing with 0)
        df = df.reindex(columns=train_columns, fill_value=0)
    
    # Scale ONLY numerical columns (not one-hot binary columns)
    if scaler is None:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    X = df.values.astype(np.float32)
    
    return X, y, train_columns, scaler, numerical_cols


# =============================================================================
# Pipeline C: CIC-IDS-2017 Preprocessing (used in all reported experiments)
# =============================================================================
def load_and_preprocess_cicids2017(data_dir, subsample_size=500000, seed=42):
    """Preprocessing for CIC-IDS-2017 dataset.
    
    Steps:
      1. Load & merge all CSVs
      2. Clean column names + labels
      3. Remove Inf/NaN rows
      4. Remove duplicate rows (CIC-IDS-2017 has ~11% duplicates, 24% in attacks)
      5. Stratified subsample
      6. Drop zero-variance columns
      7. Drop highly correlated features (>0.95 Pearson)
    
    NOTE: Returns UNSCALED data. Scaling is deferred to FLDataset so the
          scaler is fit on the training split only (no data leakage).
    
    Args:
        data_dir: Path to directory containing CIC-IDS-2017 CSV files.
        subsample_size: Number of rows to sample to keep training manageable.
        seed: Random seed for subsampling.
        
    Returns:
        X (unscaled), y, feature_columns
    """
    print(f"[CIC-IDS-2017] Loading from {data_dir}...")
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        raise ValueError(f"No CSV files found in {data_dir}")
        
    df_list = []
    for f in all_files:
        df = pd.read_csv(f, low_memory=False, encoding='latin1')
        df_list.append(df)
        
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()
    
    # Binary label mapping
    label_col = df.columns[-1]
    df[label_col] = df[label_col].astype(str).str.strip()
    df['label'] = df[label_col].apply(lambda x: 0 if x == 'BENIGN' else 1)
    df = df.drop(columns=[label_col])
    
    # Force numeric and drop NaNs/Infs
    for col in df.columns:
        if col != 'label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before_na = len(df)
    df.dropna(inplace=True)
    print(f"[CIC-IDS-2017] Dropped {before_na - len(df)} rows with NaN/Inf")
    
    # Remove duplicate rows — critical for CIC-IDS-2017
    before_dup = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[CIC-IDS-2017] Dropped {before_dup - len(df)} duplicate rows "
          f"({(before_dup - len(df)) / before_dup * 100:.1f}%)")
    
    # Subsample AFTER dedup to preserve diversity
    if subsample_size is not None and len(df) > subsample_size:
        print(f"[CIC-IDS-2017] Subsampling {len(df)} rows down to {subsample_size} (STRATIFIED)...")
        frac = subsample_size / len(df)
        df = df.groupby('label').sample(frac=frac, random_state=seed)
        
    y = df['label'].values.astype(np.longlong)
    df = df.drop('label', axis=1)
    
    # Drop zero-variance columns
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index.tolist()
    if cols_to_drop:
        print(f"[CIC-IDS-2017] Dropping {len(cols_to_drop)} zero-variance columns")
        df = df.drop(columns=cols_to_drop)
    
    # Drop highly correlated features (Pearson > 0.95)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    if corr_to_drop:
        print(f"[CIC-IDS-2017] Dropping {len(corr_to_drop)} highly correlated features")
        df = df.drop(columns=corr_to_drop)
    
    feature_columns = df.columns.tolist()
    X = df.values.astype(np.float32)
    
    benign_count = (y == 0).sum()
    attack_count = (y == 1).sum()
    print(f"[CIC-IDS-2017] Final: {len(X)} samples | {len(feature_columns)} features | "
          f"BENIGN={benign_count} ({benign_count/len(X)*100:.1f}%) | "
          f"ATTACK={attack_count} ({attack_count/len(X)*100:.1f}%)")
    
    return X, y, feature_columns


# =============================================================================
# Unified FL Dataset (with validation split)
# =============================================================================
class FLDataset:
    """Federated Learning dataset with selectable preprocessing pipeline.
    
    Splits the original training data into train + validation.
    Validation is used for model selection (checkpointing).
    Test set is used ONLY for final evaluation.
    """
    
    def __init__(self, data_path_train, data_path_test=None, dataset_type='nsl-kdd', model_type='mlp',
                 val_ratio=0.1, seed=42, subsample_size=500000):
        """
        Args:
            data_path_train: Path to NSL-KDD train file OR CIC-IDS directory.
            data_path_test: Path to NSL-KDD test file. Support None for datasets that handle splitting globally.
            dataset_type: 'nsl-kdd' or 'cicids2017'.
            model_type: 'mlp' for one-hot pipeline, 'cnn_lstm' for label-encoded pipeline.
            val_ratio: Fraction of training data held out for validation.
            seed: Random seed for reproducible train/val split.
            subsample_size: Maximum rows to keep for CIC-IDS-2017.
        """
        self.dataset_type = dataset_type
        self.model_type = model_type
        
        if dataset_type == 'nsl-kdd':
            if model_type == 'mlp':
                X_full, y_full, self.train_columns, self.scaler, _ = load_and_preprocess_mlp(data_path_train)
                X_test, y_test, _, _, _ = load_and_preprocess_mlp(
                    data_path_test, train_columns=self.train_columns, scaler=self.scaler
                )
            elif model_type == 'cnn_lstm':
                X_full, y_full, self.label_encoders, self.scaler = load_and_preprocess_cnn_lstm(data_path_train)
                X_test, y_test, _, _ = load_and_preprocess_cnn_lstm(
                    data_path_test, label_encoders=self.label_encoders, scaler=self.scaler
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}. Use 'mlp' or 'cnn_lstm'.")
            
            self.X_full = X_full
            self.y_full = y_full
            self.X_test_original = X_test
            self.y_test = y_test
            
        elif dataset_type == 'cicids2017':
            X_all, y_all, self.train_columns = load_and_preprocess_cicids2017(
                data_path_train, subsample_size=subsample_size, seed=seed
            )
            # Create a fixed test set (20%) before FL train/val splitting
            X_full, X_test, y_full, y_test = train_test_split(
                X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
            )
            
            # Fit scaler on TRAINING split only (no data leakage)
            self.scaler = StandardScaler()
            X_full = self.scaler.fit_transform(X_full)
            X_test = self.scaler.transform(X_test)
            
            self.X_full = X_full
            self.y_full = y_full
            self.X_test_original = X_test
            self.y_test = y_test
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
        # Save full data to enable reshuffling logic
        self.X_full = X_full
        self.y_full = y_full
        
        # ---------------------------------------------------------------------
        # Global Class Weights (for imbalanced data like CIC-IDS-2017)
        # Prevents clients with non-IID batches from having explosive gradients
        # ---------------------------------------------------------------------
        unique, counts = np.unique(self.y_full, return_counts=True)
        total = len(self.y_full)
        weight_vals = [total / (len(unique) * c) for c in counts]
        
        full_weights = torch.ones(2, dtype=torch.float32)
        for cls, w in zip(unique, weight_vals):
            full_weights[cls] = w
        self.global_class_weights = full_weights
        
        self.val_ratio = val_ratio
        
        self.X_test_original = X_test
        self.y_test = y_test
        
        self.feature_selector = None
        
        # Initialize normal split
        self.reshuffle(seed)
        
        print(f"[Data] Pipeline: {model_type} | Features: {self.input_dim} | "
              f"Train: {len(self.X_train)} | Val: {len(self.X_val)} | Test: {len(self.X_test)}")
        
    def reshuffle(self, seed):
        """Re-create the train/val split using a new seed (stratified train/validation split)."""
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_full, self.y_full, test_size=self.val_ratio, stratify=self.y_full, random_state=seed
        )
        
        self.X_train_raw = X_train
        self.X_val_raw = X_val
        self.X_test_raw = self.X_test_original
        
        self.y_train = y_train
        self.y_val = y_val
        
        if self.feature_selector is not None:
            self.apply_feature_selector(self.feature_selector)
        else:
            self.X_train = self.X_train_raw
            self.X_val = self.X_val_raw
            self.X_test = self.X_test_raw
            self.input_dim = self.X_train.shape[1]
            
    def apply_feature_selector(self, selector):
        """Applies a fitted feature selector to train/val/test."""
        self.feature_selector = selector
        self.X_train = selector.transform(self.X_train_raw)
        self.X_val = selector.transform(self.X_val_raw)
        self.X_test = selector.transform(self.X_test_raw)
        self.input_dim = self.X_train.shape[1]
        
    def split_data(self, num_clients, iid=True, dirichlet_alpha=0.5):
        """Splits the training data among multiple clients.

        Args:
            num_clients: Number of federated clients.
            iid: If True, random uniform shuffle (IID). If False, Dirichlet
                 allocation for realistic Non-IID label skew.
            dirichlet_alpha: Concentration parameter for Dirichlet distribution.
                Lower = more heterogeneous. 0.5 is a common choice in FL studies.
        """
        if iid:
            indices = np.random.permutation(len(self.X_train))
            client_indices = np.array_split(indices, num_clients)
        else:
            client_indices = self._dirichlet_split(num_clients, dirichlet_alpha)

        # Log per-client class distribution for verification
        for k, idx in enumerate(client_indices):
            y_k = self.y_train[idx]
            n0 = int((y_k == 0).sum())
            n1 = int((y_k == 1).sum())
            print(f"  Client {k}: {len(idx):>5d} samples | "
                  f"class0={n0:>5d} ({n0/len(idx)*100:5.1f}%) | "
                  f"class1={n1:>5d} ({n1/len(idx)*100:5.1f}%)")

        return [(self.X_train[idx], self.y_train[idx]) for idx in client_indices]

    def _dirichlet_split(self, num_clients, alpha):
        """Partition data using Dirichlet distribution for label skew.

        For each class, draws a proportion vector from Dir(alpha) and
        allocates that class's samples accordingly.  Every client is
        guaranteed at least 1 sample from each class via a minimum
        allocation floor.
        """
        num_classes = len(np.unique(self.y_train))
        label_indices = [np.where(self.y_train == c)[0] for c in range(num_classes)]

        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            np.random.shuffle(label_indices[c])
            n_c = len(label_indices[c])

            # Draw proportions from Dirichlet
            proportions = np.random.dirichlet([alpha] * num_clients)

            # Ensure every client gets at least 1 sample of this class
            min_per_client = 1
            reserved = min_per_client * num_clients
            if reserved < n_c:
                remaining = n_c - reserved
                counts = np.array([min_per_client] * num_clients)
                counts = counts + (proportions * remaining).astype(int)
            else:
                # Edge case: fewer samples than clients — just spread evenly
                counts = np.array([max(1, n_c // num_clients)] * num_clients)

            # Fix rounding so total matches exactly
            diff = n_c - counts.sum()
            for i in range(abs(diff)):
                if diff > 0:
                    counts[i % num_clients] += 1
                elif diff < 0:
                    # Find a client with > 1 to subtract from
                    idx_to_dec = (i % num_clients)
                    while counts[idx_to_dec] <= 1:
                        idx_to_dec = (idx_to_dec + 1) % num_clients
                    counts[idx_to_dec] -= 1

            # Assign indices
            start = 0
            for k in range(num_clients):
                end = start + counts[k]
                client_indices[k].extend(label_indices[c][start:end].tolist())
                start = end

        # Shuffle within each client and convert to numpy arrays
        for k in range(num_clients):
            np.random.shuffle(client_indices[k])
            client_indices[k] = np.array(client_indices[k])

        return client_indices

    def get_val_loader(self, batch_size=64):
        """Validation loader — used for model selection / checkpointing."""
        dataset = TensorDataset(torch.tensor(self.X_val), torch.tensor(self.y_val))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def get_test_loader(self, batch_size=64):
        """Test loader — used ONLY for final evaluation, never for decisions."""
        dataset = TensorDataset(torch.tensor(self.X_test), torch.tensor(self.y_test))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    print("=" * 50)
    print("Testing MLP preprocessing pipeline")
    print("=" * 50)
    dataset_mlp = FLDataset("data/KDDTrain+.txt", "data/KDDTest+.txt", model_type='mlp')
    clients = dataset_mlp.split_data(5)
    for i, (X, y) in enumerate(clients):
        print(f"  Client {i}: {len(X)} samples, features={X.shape[1]}")

    print()
    print("=" * 50)
    print("Testing CNN-LSTM preprocessing pipeline")
    print("=" * 50)
    dataset_cnn = FLDataset("data/KDDTrain+.txt", "data/KDDTest+.txt", model_type='cnn_lstm')
    clients = dataset_cnn.split_data(5)
    for i, (X, y) in enumerate(clients):
        print(f"  Client {i}: {len(X)} samples, features={X.shape[1]}")
