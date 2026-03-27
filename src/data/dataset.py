import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
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

SUPPORTED_DATASETS = ['breakhis', 'wbcd', 'seer', 'cbis_ddsm']
DATASET_NAMES = {
    'breakhis': 'BreakHis Histopathology',
    'wbcd': 'Wisconsin Breast Cancer',
    'seer': 'SEER Breast Cancer',
    'cbis_ddsm': 'CBIS-DDSM Mammography',
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


def _build_group_label_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a patient-level label table for stratified group splits.

    If a patient appears with mixed labels, use the majority label.
    """
    group_df = (
        df.groupby('patient_id')['label']
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )
    return group_df


def stratified_group_holdout_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a patient-grouped, class-stratified holdout split.
    """
    group_df = _build_group_label_table(df)

    train_groups, test_groups = train_test_split(
        group_df['patient_id'].values,
        test_size=test_size,
        random_state=seed,
        stratify=group_df['label'].values,
    )

    train_mask = df['patient_id'].isin(set(train_groups))
    test_mask = df['patient_id'].isin(set(test_groups))

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx

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
        
    # Split 1: 70% Train, 30% Temp (Val + Test), patient-grouped and stratified
    train_idx, temp_idx = stratified_group_holdout_split(df, test_size=0.30, seed=42)
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)
    
    # Split 2: 50% Val, 50% Test from the Temp partition (resulting in 15% / 15% overall)
    val_idx, test_idx = stratified_group_holdout_split(df_temp, test_size=0.50, seed=42)
    
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

    # ── Hold out 15% as a fixed test set (patient-grouped + stratified) ──
    trainval_idx, test_idx = stratified_group_holdout_split(df, test_size=0.15, seed=seed)

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
    train_tf, val_tf = get_transforms()

    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(
            df_trainval,
            df_trainval['label'],
            groups=df_trainval['patient_id'],
        )
    else:
        print("Warning: StratifiedGroupKFold not available; falling back to GroupKFold.")
        splitter = GroupKFold(n_splits=n_folds)
        split_iter = splitter.split(
            df_trainval,
            df_trainval['label'],
            groups=df_trainval['patient_id'],
        )

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
        df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

        # Verify no patient leakage within fold
        fold_train_p = set(df_train['patient_id'])
        fold_val_p = set(df_val['patient_id'])
        assert len(fold_train_p & fold_val_p) == 0, f"Leakage in fold {fold_idx}!"

        print(f"Fold {fold_idx + 1}/{n_folds} — Train: {len(df_train)} | Val: {len(df_val)}")
        print(f"  Train labels: {df_train['label'].value_counts().sort_index().to_dict()}")
        print(f"  Val labels:   {df_val['label'].value_counts().sort_index().to_dict()}")

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
# CBIS-DDSM Mammography Dataset Loader
# ─────────────────────────────────────────────

class CBISDDSMDataset(Dataset):
    """
    Dataset loader for CBIS-DDSM Breast Cancer Image Dataset.
    
    CBIS-DDSM (Curated Breast Imaging Subset of DDSM) contains mammography images
    with region of interest (ROI) masks and clinical metadata.
    
    Dataset structure:
    - Images in DICOM or PNG format
    - CSV metadata with labels (benign/malignant)
    - Both mass and calcification types
    
    This loader expects:
    - images_dir: Path to image directory
    - csv_path: Path to metadata CSV with columns: 'image_path', 'label'
    
    Auto-download from Kaggle:
    - Install kaggle: pip install kaggle
    - Set KAGGLE_USERNAME and KAGGLE_KEY environment variables
    - Dataset will auto-download if not found
    """
    
    def __init__(
        self,
        images_dir: str,
        csv_path: str,
        transform: Optional[transforms.Compose] = None,
        subset_size: Optional[int] = None,
        seed: int = 42,
    ):
        self.images_dir = images_dir
        self.csv_path = csv_path
        self.transform = transform
        
        # Load metadata
        self.df = self._load_metadata(csv_path, images_dir, subset_size, seed)
        
    def _load_metadata(
        self, 
        csv_path: str, 
        images_dir: str, 
        subset_size: Optional[int],
        seed: int
    ) -> pd.DataFrame:
        """Load and validate CBIS-DDSM metadata."""
        df = pd.read_csv(csv_path)
        
        # Standardize column names
        if 'path' in df.columns:
            df = df.rename(columns={'path': 'image_path'})
        if 'filename' in df.columns:
            df = df.rename(columns={'filename': 'image_path'})
        if 'class' in df.columns or 'diagnosis' in df.columns or 'type' in df.columns:
            label_col = 'class' if 'class' in df.columns else ('diagnosis' if 'diagnosis' in df.columns else 'type')
            df = df.rename(columns={label_col: 'label'})
        
        # Map labels to binary if needed
        if 'label' in df.columns and df['label'].dtype == 'object':
            label_map = {
                'benign': 0, 'BENIGN': 0, 'Benign': 0, 'B': 0,
                'malignant': 1, 'MALIGNANT': 1, 'Malignant': 1, 'M': 1,
                'normal': 0, 'Normal': 0, 'N': 0,
                'abnormal': 1, 'Abnormal': 1, 'A': 1,
            }
            df['label'] = df['label'].map(label_map).fillna(1)  # Default to malignant if unknown
        
        # Ensure image paths are relative to images_dir if needed
        if not df['image_path'].iloc[0].startswith(images_dir):
            df['image_path'] = df['image_path'].apply(
                lambda x: os.path.join(images_dir, x) if not os.path.isabs(x) else x
            )
        
        # Filter out non-existent files
        df = df[df['image_path'].apply(os.path.exists)]
        
        if subset_size:
            df = df.sample(n=min(subset_size, len(df)), random_state=seed)
        
        return df.reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = int(row['label'])
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback: create dummy image
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def download_cbis_ddsm(data_dir: str, kaggle_username: Optional[str] = None, kaggle_key: Optional[str] = None) -> bool:
    """
    Auto-download CBIS-DDSM dataset from Kaggle.
    
    Args:
        data_dir: Directory to download dataset to
        kaggle_username: Kaggle username (or set KAGGLE_USERNAME env var)
        kaggle_key: Kaggle API key (or set KAGGLE_KEY env var)
    
    Returns:
        True if download successful, False otherwise
    """
    import subprocess
    import os
    
    # Get credentials
    username = kaggle_username or os.environ.get('KAGGLE_USERNAME')
    key = kaggle_key or os.environ.get('KAGGLE_KEY')
    
    if not username or not key:
        print("⚠️  Kaggle credentials not found!")
        print("Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables,")
        print("or provide credentials in config.yaml")
        print("\nTo get API key: https://www.kaggle.com/account")
        return False
    
    # Set credentials as env vars if provided
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    
    try:
        print("📥 Downloading CBIS-DDSM dataset from Kaggle...")
        print("   This may take 10-30 minutes depending on connection speed.")
        
        # Create directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Download dataset
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'awsaf49/cbis-ddsm-breast-cancer-image-dataset'],
            cwd=data_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Download failed: {result.stderr}")
            return False
        
        print("✅ Download complete. Extracting...")
        
        # Extract
        import zipfile
        zip_path = os.path.join(data_dir, 'cbis-ddsm-breast-cancer-image-dataset.zip')
        
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_path)
            print("✅ Extraction complete!")
            return True
        else:
            print("⚠️  Downloaded file not found, but download reported success.")
            return True
            
    except FileNotFoundError:
        print("❌ Kaggle CLI not found. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def prepare_cbis_ddsm(data_dir: str, subset_size: Optional[int] = None, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Prepare CBIS-DDSM dataset with train/val/test splits.
    
    Args:
        data_dir: Base directory for CBIS-DDSM
        subset_size: Optional subset size for testing
        seed: Random seed
    
    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """
    import json
    
    # Paths
    images_dir = os.path.join(data_dir, 'images')
    csv_path = os.path.join(data_dir, 'metadata.csv')
    split_info_path = os.path.join(data_dir, 'splits.json')
    
    # Check if metadata exists
    if not os.path.exists(csv_path):
        # Try to find CSV in subdirectory
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.csv') and 'metadata' in file.lower():
                    csv_path = os.path.join(root, file)
                    break
    
    # Load full metadata
    df = pd.read_csv(csv_path)
    
    # Standardize columns
    if 'path' in df.columns:
        df = df.rename(columns={'path': 'image_path'})
    if 'filename' in df.columns:
        df = df.rename(columns={'filename': 'image_path'})
    if 'class' in df.columns or 'diagnosis' in df.columns or 'type' in df.columns:
        label_col = 'class' if 'class' in df.columns else ('diagnosis' if 'diagnosis' in df.columns else 'type')
        df = df.rename(columns={label_col: 'label'})
    
    # Map labels
    if 'label' in df.columns and df['label'].dtype == 'object':
        label_map = {
            'benign': 0, 'BENIGN': 0, 'Benign': 0, 'B': 0,
            'malignant': 1, 'MALIGNANT': 1, 'Malignant': 1, 'M': 1,
        }
        df['label'] = df['label'].map(label_map).fillna(1)
    
    # Ensure paths
    if not df['image_path'].iloc[0].startswith(images_dir):
        df['image_path'] = df['image_path'].apply(
            lambda x: os.path.join(images_dir, x) if not os.path.isabs(x) else x
        )
    
    # Filter existing files
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    
    print(f"CBIS-DDSM: Found {len(df)} valid images")
    print(df['label'].value_counts())
    
    if subset_size:
        df = df.sample(n=min(subset_size, len(df)), random_state=seed).reset_index(drop=True)
    
    # Split: 70% train, 15% val, 15% test
    from sklearn.model_selection import train_test_split
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df['image_path'], df['label'], test_size=0.15, random_state=seed, stratify=df['label']
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=seed, stratify=y_trainval  # 0.176 ≈ 15/85
    )
    
    df_train = pd.DataFrame({'image_path': X_train, 'label': y_train})
    df_val = pd.DataFrame({'image_path': X_val, 'label': y_val})
    df_test = pd.DataFrame({'image_path': X_test, 'label': y_test})
    
    # Save splits
    split_info = {
        'train': len(df_train),
        'val': len(df_val),
        'test': len(df_test),
        'total': len(df)
    }
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    df_train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    
    print(f"Split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    
    # Create dataloaders
    train_tf, val_tf = get_transforms()
    
    train_ds = CBISDDSMDataset(images_dir, os.path.join(data_dir, 'train.csv'), transform=train_tf)
    val_ds = CBISDDSMDataset(images_dir, os.path.join(data_dir, 'val.csv'), transform=val_tf)
    test_ds = CBISDDSMDataset(images_dir, os.path.join(data_dir, 'test.csv'), transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, 2


def get_cbis_ddsm_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    subset_size: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42,
    auto_download: bool = True,
    kaggle_username: Optional[str] = None,
    kaggle_key: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create DataLoaders for CBIS-DDSM dataset with auto-download.
    
    Args:
        data_dir: Directory for CBIS-DDSM dataset
        batch_size: Batch size
        subset_size: Optional subset size
        num_workers: DataLoader workers
        seed: Random seed
        auto_download: Whether to auto-download if not found
        kaggle_username: Kaggle username (optional)
        kaggle_key: Kaggle API key (optional)
    
    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """
    set_seed(seed)
    
    # Check if dataset exists
    images_dir = os.path.join(data_dir, 'images')
    csv_path = os.path.join(data_dir, 'metadata.csv')
    
    dataset_exists = (
        os.path.exists(images_dir) and 
        os.path.exists(csv_path) and
        len(os.listdir(images_dir)) > 0
    )
    
    if not dataset_exists:
        if auto_download:
            print("⚠️  CBIS-DDSM dataset not found. Attempting auto-download...")
            success = download_cbis_ddsm(data_dir, kaggle_username, kaggle_key)
            
            if not success:
                raise RuntimeError(
                    "Failed to download CBIS-DDSM. Please:\n"
                    "1. Install kaggle: pip install kaggle\n"
                    "2. Set KAGGLE_USERNAME and KAGGLE_KEY env vars\n"
                    "3. Or manually download from: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset"
                )
        else:
            raise FileNotFoundError(
                f"CBIS-DDSM dataset not found at {data_dir}. "
                "Set auto_download=True or download manually from Kaggle."
            )
    
    # Prepare and return dataloaders
    return prepare_cbis_ddsm(data_dir, subset_size, seed)


# ─────────────────────────────────────────────
# Unified Multi-Dataset Loader
# ─────────────────────────────────────────────

def get_multidataset_kfold_splits(
    dataset_name: str,
    data_dir: str,
    task: str = 'binary',
    n_folds: int = 5,
    batch_size: int = 32,
    subset_size: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Unified k-fold cross-validation loader for multiple datasets.
    
    For BreakHis: Uses patient-grouped k-fold (get_kfold_splits)
    For other datasets: Uses stratified k-fold on samples
    
    Args:
        dataset_name: 'breakhis', 'wbcd', 'seer', or 'cbis_ddsm'
        data_dir: Data directory path
        task: 'binary' or 'multi'
        n_folds: Number of folds
        batch_size: Batch size
        subset_size: Optional subset size
        num_workers: DataLoader workers
        seed: Random seed
    
    Yields:
        (fold_idx, train_loader, val_loader, test_loader, num_classes)
    """
    if dataset_name == 'breakhis':
        # Use patient-grouped k-fold for BreakHis
        yield from get_kfold_splits(
            data_dir=data_dir,
            task=task,
            n_folds=n_folds,
            batch_size=batch_size,
            subset_size=subset_size,
            num_workers=num_workers,
            seed=seed,
        )
    else:
        # For other datasets (WBCD, SEER, CBIS-DDSM), use sample-level k-fold
        from sklearn.model_selection import StratifiedKFold
        
        # Get train/val/test split first (hold out 15% for test)
        if dataset_name == 'wbcd':
            from sklearn.model_selection import train_test_split
            import pandas as pd
            
            df = pd.read_csv(data_dir) if data_dir.endswith('.csv') else pd.read_csv(os.path.join(data_dir, 'wbcd.csv'))
            
            if 'diagnosis' in df.columns:
                df['label'] = (df['diagnosis'] == 'M').astype(int)
                df = df.drop('diagnosis', axis=1)
            
            id_cols = [c for c in df.columns if c.lower() == 'id']
            if id_cols:
                df = df.drop(columns=id_cols)
            
            if subset_size:
                df = df.sample(n=min(subset_size, len(df)), random_state=seed)
            
            X = df.drop('label', axis=1).values
            y = df['label'].values
            
            # Hold out test set
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y, test_size=0.15, random_state=seed, stratify=y
            )
            
            # Create test loader
            from torch.utils.data import TensorDataset
            test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            
            # K-fold on trainval
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            train_tf, val_tf = get_transforms()
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
                X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
                y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
                
                train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
                val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
                
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
                
                yield fold_idx, train_loader, val_loader, test_loader, 2
        
        elif dataset_name == 'cbis_ddsm':
            # For CBIS-DDSM, first prepare the dataset
            train_loader_all, val_loader_all, test_loader, num_classes = get_cbis_ddsm_dataloaders(
                data_dir=data_dir,
                batch_size=batch_size,
                subset_size=subset_size,
                num_workers=num_workers,
                seed=seed,
            )
            
            # Now create k-fold splits from the training data
            # Note: This is a simplified version - for proper CV, you'd want to reload data for each fold
            for fold_idx in range(n_folds):
                # For now, reuse the same train/val splits
                # TODO: Implement proper k-fold for CBIS-DDSM with different splits per fold
                yield fold_idx, train_loader_all, val_loader_all, test_loader, num_classes
        
        elif dataset_name == 'seer':
            # Similar to WBCD
            from sklearn.model_selection import train_test_split
            import pandas as pd
            
            df = pd.read_csv(data_dir) if data_dir.endswith('.csv') else pd.read_csv(os.path.join(data_dir, 'seer.csv'))
            
            target_column = 'VitalStatusRecoded'
            if target_column in df.columns:
                if df[target_column].dtype == 'object':
                    df['label'] = (df[target_column] == 'Alive').astype(int)
                else:
                    df['label'] = df[target_column]
                df = df.drop(target_column, axis=1)
            
            df = df.fillna(df.median(numeric_only=True)).dropna()
            
            if subset_size:
                df = df.sample(n=min(subset_size, len(df)), random_state=seed)
            
            X = df.drop('label', axis=1).values
            y = df['label'].values
            
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y, test_size=0.15, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
            )
            
            from torch.utils.data import TensorDataset
            test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
                X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
                y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
                
                train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
                val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
                
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
                
                yield fold_idx, train_loader, val_loader, test_loader, 2
