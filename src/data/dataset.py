import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import pandas as pd
from typing import Optional, Tuple, Dict

def set_seed(seed=42):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Multi-Dataset Support
# ─────────────────────────────────────────────

SUPPORTED_DATASETS = ['breakhis', 'wbcd', 'seer']
DATASET_NAMES = {
    'breakhis': 'BreakHis Histopathology',
    'wbcd': 'Wisconsin Breast Cancer',
    'seer': 'SEER Breast Cancer',
}

class BreakHisDataset(Dataset):
    """
    Dataset loader for BreakHis breast cancer dataset.
    BreakHis directory structure usually resembles:
    BreaKHis_v1/histology_slides/breast/{benign|malignant}/SOB/{tumor_type}/{patient_id}/{magnification}/...
    """
    def __init__(self, df_paths, df_labels, transform=None):
        self.paths = df_paths
        self.labels = df_labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths.iloc[idx]
        label = self.labels.iloc[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Instead of crashing, load a dummy or just re-raise
            raise e
            
        if self.transform:
            image = self.transform(image)
            
        return image, int(label)

def get_transforms():
    """Returns train and validation/test transforms."""
    # Using robust augmentations for medical images as discussed
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def parse_breakhis_directory(data_dir, task="binary"):
    """
    Walks the BreakHis directory and extracts paths, labels, and patient IDs.
    task: "binary" (Benign vs Malignant) or "multi" (IDC, ILC, Fibroadenoma).
    """
    paths = []
    labels = []
    patient_ids = []
    
    # Standard classes for 3-class targeted task
    multi_target_map = {
        'IDC': 0, # Invasive Ductal Carcinoma
        'ILC': 1, # Invasive Lobular Carcinoma
        'F': 2    # Fibroadenoma
    }

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                
                # Extract metadata from path
                # Example path segment: .../malignant/SOB/ductal_carcinoma/SOB_M_DC_14-11031/...
                # Note: BreakHis file/folder naming conventions vary slightly, 
                # but Patient ID usually takes the form SOB_B_F_14-14134 
                
                parts = full_path.replace("\\", "/").split('/')
                
                # Infer binary class
                if 'benign' in full_path.lower():
                    binary_label = 0
                elif 'malignant' in full_path.lower():
                    binary_label = 1
                else:
                    continue # Skip if cant determine
                    
                # Infer specific type for multi-class
                # The tumor type is usually represented by acronyms in the filename (e.g., SOB_M_DC)
                # or in the folder structure (e.g. ductal_carcinoma)
                file_upper = file.upper()
                multi_label = -1
                if 'DC' in file_upper: # Ductal Carcinoma
                    multi_label = multi_target_map['IDC']
                elif 'LC' in file_upper: # Lobular Carcinoma
                    multi_label = multi_target_map['ILC']
                elif '_F_' in file_upper: # Fibroadenoma
                    multi_label = multi_target_map['F']
                
                # Extract patient ID (usually the 3rd element in the hyphenated filename or folder)
                # e.g., SOB_M_DC_14-9461_100X_... -> Patient ID is '14-9461'
                try:
                    patient_id = file.split('_')[3] # Assuming standard format
                except IndexError:
                    # Fallback if filename format is unexpected
                    patient_id = os.path.basename(os.path.dirname(os.path.dirname(full_path)))
                
                if task == "binary":
                    paths.append(full_path)
                    labels.append(binary_label)
                    patient_ids.append(patient_id)
                elif task == "multi" and multi_label != -1: # Only include if it's one of the 3 target classes
                    paths.append(full_path)
                    labels.append(multi_label)
                    patient_ids.append(patient_id)
                    
    df = pd.DataFrame({
        'path': paths,
        'label': labels,
        'patient_id': patient_ids
    })
    
    return df

def get_dataloaders(data_dir, task="binary", batch_size=32, subset_size=None, num_workers=4):
    """
    Creates DataLoaders with Grouped Stratification by Patient ID (70/15/15 split).
    """
    df = parse_breakhis_directory(data_dir, task)
    
    if len(df) == 0:
        raise ValueError(f"No images found in {data_dir} for task {task}.")
    
    if subset_size:
        # Take a subset but try to keep it patient-grouped if possible
        # We sample randomly so we get a mix of classes, which is crucial for
        # ROC AUC scoring not to crash due to single-class batches.
        df = df.sample(n=min(subset_size, len(df)), random_state=42).reset_index(drop=True)
        print(f"Using random subset of size: {len(df)}")
        
    print(f"Total images found: {len(df)} | Task: {task}")
    print(df['label'].value_counts())
        
    # Split 1: 70% Train, 30% Temp (Val + Test)
    gss_train = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, temp_idx = next(gss_train.split(df, df['label'], groups=df['patient_id']))
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)
    
    # Split 2: 50% Val, 50% Test from the Temp partition (resulting in 15% / 15% overall)
    gss_test = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_test.split(df_temp, df_temp['label'], groups=df_temp['patient_id']))
    
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)
    
    # Assertions to ensure NO data leakage (No overlapping patients)
    train_patients = set(df_train['patient_id'])
    val_patients = set(df_val['patient_id'])
    test_patients = set(df_test['patient_id'])
    
    assert len(train_patients.intersection(val_patients)) == 0, "Leakage: Train and Val share patients!"
    assert len(train_patients.intersection(test_patients)) == 0, "Leakage: Train and Test share patients!"
    assert len(val_patients.intersection(test_patients)) == 0, "Leakage: Val and Test share patients!"

    print(f"Split completed. Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    
    train_tf, val_tf = get_transforms()
    
    train_ds = BreakHisDataset(df_train['path'], df_train['label'], transform=train_tf)
    val_ds = BreakHisDataset(df_val['path'], df_val['label'], transform=val_tf)
    test_ds = BreakHisDataset(df_test['path'], df_test['label'], transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, len(df['label'].unique())


def get_kfold_splits(data_dir, task="binary", n_folds=5, batch_size=32,
                     subset_size=None, num_workers=4, seed=42):
    """
    Generator that yields (fold_idx, train_loader, val_loader) for K-Fold CV.

    Uses GroupKFold to ensure patient-level separation between train/val.
    A held-out test set (15% of data) is reserved BEFORE folding.

    Args:
        data_dir: Path to BreakHis root directory.
        task: "binary" or "multi".
        n_folds: Number of CV folds (default: 5).
        batch_size: Batch size for DataLoaders.
        subset_size: Optional int to limit dataset size for quick tests.
        num_workers: DataLoader workers.
        seed: Random seed.

    Yields:
        (fold_idx, train_loader, val_loader, test_loader, num_classes)
        test_loader is the same across all folds (held-out).
    """
    set_seed(seed)
    df = parse_breakhis_directory(data_dir, task)

    if len(df) == 0:
        raise ValueError(f"No images found in {data_dir} for task {task}.")

    if subset_size:
        df = df.sample(n=min(subset_size, len(df)), random_state=seed).reset_index(drop=True)
        print(f"Using random subset of size: {len(df)}")

    num_classes = len(df['label'].unique())
    print(f"Total images: {len(df)} | Task: {task} | Classes: {num_classes}")
    print(df['label'].value_counts())

    # ── Hold out 15% as a fixed test set (patient-grouped) ──
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, df['label'], groups=df['patient_id']))

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Verify no patient leakage in test set
    tv_patients = set(df_trainval['patient_id'])
    test_patients = set(df_test['patient_id'])
    assert len(tv_patients & test_patients) == 0, "Leakage: TrainVal and Test share patients!"

    print(f"Held-out test set: {len(df_test)} images ({len(test_patients)} patients)")
    print(f"Train+Val pool: {len(df_trainval)} images — splitting into {n_folds} folds\n")

    # ── Build fixed test loader ──
    _, val_tf = get_transforms()
    test_ds = BreakHisDataset(df_test['path'], df_test['label'], transform=val_tf)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # ── K-Fold on train+val pool ──
    gkf = GroupKFold(n_splits=n_folds)
    train_tf, val_tf = get_transforms()

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(df_trainval, df_trainval['label'], groups=df_trainval['patient_id'])
    ):
        df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
        df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

        # Verify no patient leakage within fold
        fold_train_p = set(df_train['patient_id'])
        fold_val_p = set(df_val['patient_id'])
        assert len(fold_train_p & fold_val_p) == 0, f"Leakage in fold {fold_idx}!"

        print(f"Fold {fold_idx + 1}/{n_folds} — Train: {len(df_train)} | Val: {len(df_val)}")

        train_ds = BreakHisDataset(df_train['path'], df_train['label'], transform=train_tf)
        val_ds = BreakHisDataset(df_val['path'], df_val['label'], transform=val_tf)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        yield fold_idx, train_loader, val_loader, test_loader, num_classes


# ─────────────────────────────────────────────
# Wisconsin Breast Cancer Dataset (WBCD) Loader
# ─────────────────────────────────────────────

class WBCDDataset(Dataset):
    """
    Dataset loader for Wisconsin Breast Cancer Dataset (WBCD).
    
    WBCD contains features computed from digitized images of fine needle aspirates
    of breast masses, with labels for benign and malignant cases.
    
    Features (10 real-valued):
    - radius, texture, perimeter, area, smoothness
    - compactness, concavity, concave points, symmetry, fractal dimension
    
    Label: 0 = benign, 1 = malignant
    """
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[transforms.Compose] = None,
        subset_size: Optional[int] = None,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.transform = transform
        
        # Load data
        df = self._load_wbcd(data_path)
        
        if subset_size is not None:
            df = df.sample(n=min(subset_size, len(df)), random_state=seed)
        
        self.data = df.drop('label', axis=1).values.astype(np.float32)
        self.labels = df['label'].values.astype(np.int64)
    
    def _load_wbcd(self, data_path: str) -> pd.DataFrame:
        """Load WBCD dataset from CSV."""
        df = pd.read_csv(data_path)
        
        # Handle different column naming conventions
        if 'diagnosis' in df.columns:
            # Map diagnosis to binary label
            df['label'] = (df['diagnosis'] == 'M').astype(int)
            df = df.drop('diagnosis', axis=1)
        elif 'Label' in df.columns:
            df = df.rename(columns={'Label': 'label'})
        
        # Drop ID column if present
        if 'id' in df.columns or 'ID' in df.columns:
            df = df.drop(columns=[c for c in df.columns if c.lower() == 'id'], axis=1)
        
        return df
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.data[idx].copy()
        label = self.labels[idx]
        
        # Convert to image-like format for model compatibility
        # Reshape to (1, 10, 1) and apply transform if any
        sample = sample.reshape(1, -1, 1)
        
        if self.transform:
            # For tabular data, we skip image transforms
            pass
        
        sample = torch.from_numpy(sample).float()
        # Flatten for fully connected layers
        sample = sample.view(-1)
        
        return sample, label


def get_wbcd_dataloaders(
    data_path: str,
    batch_size: int = 32,
    subset_size: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create DataLoaders for WBCD dataset.
    
    Args:
        data_path: Path to WBCD CSV file.
        batch_size: Batch size.
        subset_size: Optional subset size for testing.
        num_workers: DataLoader workers.
        seed: Random seed.
        val_split: Validation split ratio.
        test_split: Test split ratio.
    
    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """
    set_seed(seed)
    
    # Load full dataset
    df = pd.read_csv(data_path)
    
    if 'diagnosis' in df.columns:
        df['label'] = (df['diagnosis'] == 'M').astype(int)
        df = df.drop('diagnosis', axis=1)
    
    # Drop ID if present
    id_cols = [c for c in df.columns if c.lower() == 'id']
    if id_cols:
        df = df.drop(columns=id_cols)
    
    if subset_size:
        df = df.sample(n=min(subset_size, len(df)), random_state=seed)
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_split/(1-test_split), 
        random_state=seed, stratify=y_trainval
    )
    
    # Create simple tensor datasets
    from torch.utils.data import TensorDataset
    
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long()
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, 2


# ─────────────────────────────────────────────
# SEER Breast Cancer Dataset Loader
# ─────────────────────────────────────────────

class SEERDataset(Dataset):
    """
    Dataset loader for SEER Breast Cancer Dataset.
    
    SEER (Surveillance, Epidemiology, and End Results Program) contains
    cancer incidence and survival data from multiple registries.
    
    Typical features:
    - Age at diagnosis
    - Race/ethnicity
    - Tumor size, grade, stage
    - Survival time, vital status
    
    This loader expects preprocessed SEER data in CSV format.
    """
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[transforms.Compose] = None,
        subset_size: Optional[int] = None,
        seed: int = 42,
        target_column: str = 'VitalStatusRecoded',
    ):
        self.data_path = data_path
        self.transform = transform
        self.target_column = target_column
        
        # Load data
        df = self._load_seer(data_path, target_column)
        
        if subset_size is not None:
            df = df.sample(n=min(subset_size, len(df)), random_state=seed)
        
        # Separate features and labels
        feature_cols = [c for c in df.columns if c != 'label']
        self.data = df[feature_cols].values.astype(np.float32)
        self.labels = df['label'].values.astype(np.int64)
    
    def _load_seer(self, data_path: str, target_column: str) -> pd.DataFrame:
        """Load and preprocess SEER dataset."""
        df = pd.read_csv(data_path)
        
        # Map target to binary label if needed
        if target_column in df.columns:
            if df[target_column].dtype == 'object':
                # Binary classification: Alive vs Deceased
                df['label'] = (df[target_column] == 'Alive').astype(int)
            else:
                df['label'] = df[target_column]
            df = df.drop(target_column, axis=1)
        elif 'label' not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Drop rows with remaining NaN
        df = df.dropna()
        
        return df
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.data[idx].copy()
        label = self.labels[idx]
        
        sample = torch.from_numpy(sample).float()
        return sample, label


def get_seer_dataloaders(
    data_path: str,
    batch_size: int = 32,
    subset_size: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42,
    val_split: float = 0.15,
    test_split: float = 0.15,
    target_column: str = 'VitalStatusRecoded',
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create DataLoaders for SEER dataset.
    
    Args:
        data_path: Path to SEER CSV file.
        batch_size: Batch size.
        subset_size: Optional subset size.
        num_workers: DataLoader workers.
        seed: Random seed.
        val_split: Validation split.
        test_split: Test split.
        target_column: Target column for prediction.
    
    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """
    set_seed(seed)
    
    df = pd.read_csv(data_path)
    
    # Map target
    if target_column in df.columns:
        if df[target_column].dtype == 'object':
            df['label'] = (df[target_column] == 'Alive').astype(int)
        else:
            df['label'] = df[target_column]
        df = df.drop(target_column, axis=1)
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    df = df.dropna()
    
    if subset_size:
        df = df.sample(n=min(subset_size, len(df)), random_state=seed)
    
    # Split
    from sklearn.model_selection import train_test_split
    
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    if len(np.unique(y_trainval)) > 1:
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_split/(1-test_split),
            random_state=seed, stratify=y_trainval
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_split/(1-test_split),
            random_state=seed
        )
    
    from torch.utils.data import TensorDataset
    
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, 2


# ─────────────────────────────────────────────
# Unified Multi-Dataset Loader
# ─────────────────────────────────────────────

def get_multidataset_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int = 32,
    subset_size: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Unified dataloader for multiple breast cancer datasets.
    
    Args:
        dataset_name: 'breakhis', 'wbcd', or 'seer'.
        data_dir: Data directory path.
        batch_size: Batch size.
        subset_size: Optional subset size.
        num_workers: DataLoader workers.
        seed: Random seed.
    
    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """
    if dataset_name == 'breakhis':
        return get_dataloaders(
            data_dir=data_dir,
            task='binary',
            batch_size=batch_size,
            subset_size=subset_size,
            num_workers=num_workers,
        )
    elif dataset_name == 'wbcd':
        data_path = os.path.join(data_dir, 'wbcd.csv')
        if not os.path.exists(data_path):
            data_path = data_dir  # Try direct path
        return get_wbcd_dataloaders(
            data_path=data_path,
            batch_size=batch_size,
            subset_size=subset_size,
            num_workers=num_workers,
            seed=seed,
        )
    elif dataset_name == 'seer':
        data_path = os.path.join(data_dir, 'seer.csv')
        if not os.path.exists(data_path):
            data_path = data_dir
        return get_seer_dataloaders(
            data_path=data_path,
            batch_size=batch_size,
            subset_size=subset_size,
            num_workers=num_workers,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Choose from: {SUPPORTED_DATASETS}")

