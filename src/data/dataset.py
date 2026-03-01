import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import pandas as pd

def set_seed(seed=42):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

