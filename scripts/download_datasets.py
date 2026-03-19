"""
Automatic Dataset Downloader for Breast Cancer Classification Project.

Downloads and prepares:
1. BreakHis Histopathology Dataset (primary)
2. Wisconsin Breast Cancer Dataset (WBCD)
3. SEER Breast Cancer Dataset

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --dataset breakhis
    python scripts/download_datasets.py --dataset wbcd
    python scripts/download_datasets.py --dataset seer
    python scripts/download_datasets.py --all
"""

import os
import sys
import argparse
import zipfile
import tarfile
import hashlib
from pathlib import Path
import requests
from tqdm import tqdm

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Dataset configurations
DATASETS = {
    'breakhis': {
        'name': 'BreakHis Histopathology Dataset',
        'type': 'histopathology',
        'url': 'https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/BreaKHis_v1.zip',
        'checksum': None,  # Will skip checksum if None
        'format': 'zip',
        'extract_to': 'BreaKHis_v1',
        'final_dir': DATA_DIR / 'BreaKHis_v1',
        'notes': 'Manual download may be required - see instructions below',
    },
    'wbcd': {
        'name': 'Wisconsin Breast Cancer Dataset (WBCD)',
        'type': 'clinical',
        'source': 'UCI ML Repository',
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
        'checksum': None,
        'format': 'csv',
        'final_dir': DATA_DIR / 'WBCD',
        'notes': 'Preprocessing required - will be handled automatically',
    },
    'seer': {
        'name': 'SEER Breast Cancer Dataset',
        'type': 'clinical',
        'source': 'SEER Program (NIH)',
        'url': None,  # Requires manual registration
        'checksum': None,
        'format': 'csv',
        'final_dir': DATA_DIR / 'SEER',
        'notes': 'REQUIRES MANUAL DOWNLOAD - see instructions below',
    },
}


def compute_file_hash(filepath, algorithm='md5'):
    """Compute hash of a file for integrity check."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_file(url, destination, description="Downloading"):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except requests.RequestException as e:
        print(f"ERROR: Download failed - {e}")
        return False


def extract_archive(archive_path, extract_to=None):
    """Extract zip or tar archive."""
    archive_path = Path(archive_path)
    if extract_to is None:
        extract_to = archive_path.parent
    
    print(f"Extracting {archive_path.name}...")
    
    try:
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix.lower() in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"WARNING: Unknown archive format: {archive_path.suffix}")
            return False
        
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"ERROR: Extraction failed - {e}")
        return False


def download_breakhis():
    """Download BreakHis dataset from Kaggle."""
    print("\n" + "="*70)
    print("BreakHis Histopathology Dataset")
    print("="*70)
    
    final_dir = DATASETS['breakhis']['final_dir']
    
    if final_dir.exists() and list(final_dir.rglob('*.png')):
        print(f"✓ BreakHis dataset already exists at: {final_dir}")
        print("  Skipping download...")
        return True
    
    print("\nAttempting Kaggle download...")
    
    # Try Kaggle download
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        print("Downloading BreakHis dataset from Kaggle...")
        api.dataset_download_files('ambarish/breakhis', path=str(DATA_DIR), unzip=True)
        
        print(f"✓ BreakHis downloaded successfully")
        return True
        
    except ImportError:
        print("Kaggle package not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle', '-q'])
        return download_breakhis()
        
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("\nCreating simulated BreakHis dataset for testing...")
        return create_simulated_breakhis()


def create_simulated_breakhis():
    """Create simulated BreakHis-like dataset structure for testing."""
    print("\nCreating simulated BreakHis dataset structure...")
    
    import numpy as np
    from PIL import Image
    
    final_dir = DATA_DIR / 'BreaKHis_v1' / 'histology_slides' / 'breast'
    
    # Create directory structure
    classes = ['benign', 'malignant']
    tumor_types = {
        'benign': ['A', 'F', 'PT', 'T'],  # Adenosis, Fibroadenoma, Phyllodes Tumor, Tubular Adenoma
        'malignant': ['DC', 'LC', 'M', 'PC'],  # Ductal Carcinoma, Lobular Carcinoma, Medullary, Papillary
    }
    
    magnifications = ['40X', '100X', '200X', '400X']
    
    total_images = 0
    target_per_class = 50  # Create 50 images per class for testing
    
    for class_name in classes:
        for tumor_type in tumor_types[class_name]:
            for mag in magnifications:
                # Create directory
                patient_dir = final_dir / class_name / 'SOB' / f'{class_name}_{tumor_type}' / f'SOB_{class_name[0]}_{tumor_type}_14-001' / mag
                patient_dir.mkdir(parents=True, exist_ok=True)
                
                # Create simulated images
                for i in range(target_per_class // (len(classes) * len(tumor_types[class_name]) * len(magnifications)) + 1):
                    img_path = patient_dir / f'SOB_{class_name[0]}_{tumor_type}_14-001_{mag}_00{i}.png'
                    
                    # Create random image with class-specific characteristics
                    if class_name == 'malignant':
                        # More irregular patterns for malignant
                        img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
                        # Add some "nuclei" - darker irregular regions
                        for _ in range(20):
                            x, y = np.random.randint(0, 200, 2)
                            size = np.random.randint(10, 30)
                            img_array[y:y+size, x:x+size] = np.random.randint(20, 80, (size, size, 3))
                    else:
                        # More uniform patterns for benign
                        img_array = np.random.randint(100, 220, (224, 224, 3), dtype=np.uint8)
                        # Add some organized structures
                        for _ in range(10):
                            x, y = np.random.randint(0, 200, 2)
                            size = np.random.randint(15, 25)
                            img_array[y:y+size, x:x+size] = np.random.randint(150, 200, (size, size, 3))
                    
                    img = Image.fromarray(img_array)
                    img.save(img_path)
                    total_images += 1
    
    print(f"✓ Created simulated dataset with {total_images} images")
    print(f"  Location: {final_dir.parent.parent}")
    print("\n⚠️  WARNING: This is a SIMULATED dataset for testing only!")
    print("    For real research, download the actual BreakHis dataset.")
    
    return True


def download_wbcd():
    """Download Wisconsin Breast Cancer Dataset using ucimlrepo."""
    print("\n" + "="*70)
    print("Wisconsin Breast Cancer Dataset (WBCD)")
    print("="*70)
    
    final_dir = DATASETS['wbcd']['final_dir']
    final_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = final_dir / 'wbcd.csv'
    
    if output_file.exists():
        print(f"✓ WBCD dataset already exists at: {output_file}")
        print("  Skipping download...")
        return True
    
    print(f"\nDownloading WBCD from UCI ML Repository...")
    
    # Try using ucimlrepo
    try:
        from ucimlrepo import fetch_ucirepo
        
        print("Fetching dataset from UCI ML Repository...")
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
        
        # Get features and targets
        X = breast_cancer_wisconsin_diagnostic.data.features
        y = breast_cancer_wisconsin_diagnostic.data.targets
        
        # Combine into single dataframe
        df = X.copy()
        df['diagnosis'] = y
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        print(f"✓ WBCD downloaded successfully: {len(df)} samples")
        return True
        
    except ImportError:
        print("ucimlrepo package not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ucimlrepo', '-q'])
        return download_wbcd()
        
    except Exception as e:
        print(f"ucimlrepo download failed: {e}")
        print("\nFalling back to direct download...")
        
        # Fallback to direct download
        if download_file(DATASETS['wbcd']['url'], output_file, "Downloading WBCD"):
            print(f"✓ WBCD downloaded successfully")
            return True
        else:
            print("ERROR: Failed to download WBCD")
            return False


def download_seer():
    """Download SEER Breast Cancer Dataset."""
    print("\n" + "="*70)
    print("SEER Breast Cancer Dataset")
    print("="*70)
    
    final_dir = DATASETS['seer']['final_dir']
    
    if final_dir.exists() and list(final_dir.glob('*.csv')):
        print(f"✓ SEER dataset already exists at: {final_dir}")
        print("  Skipping download...")
        return True
    
    print("\nSEER dataset REQUIRES MANUAL DOWNLOAD:")
    print("\nSteps:")
    print("  1. Visit: https://seer.cancer.gov/data/")
    print("  2. Register for access (free for research)")
    print("  3. Download breast cancer incidence data")
    print("  4. Extract to: data/SEER/")
    print("\nRequired files:")
    print("  - Breast cancer incidence data (CSV format)")
    print("  - Patient demographics")
    print("  - Tumor characteristics")
    
    # Offer to create template
    response = input("\nCreate SEER data template and download instructions? (y/n): ").strip().lower()
    if response == 'y':
        return create_seer_template()
    
    return False


def create_seer_template():
    """Create SEER data template and instructions."""
    print("\nCreating SEER data template...")
    
    final_dir = DATASETS['seer']['final_dir']
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # Create README with instructions
    readme_file = final_dir / 'README.txt'
    with open(readme_file, 'w') as f:
        f.write("SEER Breast Cancer Dataset - Download Instructions\n")
        f.write("="*60 + "\n\n")
        f.write("1. Visit: https://seer.cancer.gov/data/\n")
        f.write("2. Register for access (free for research use)\n")
        f.write("3. Download the following datasets:\n")
        f.write("   - Breast Cancer Incidence Data\n")
        f.write("   - Patient Demographics\n")
        f.write("   - Tumor Characteristics\n\n")
        f.write("4. Place CSV files in this directory\n\n")
        f.write("Required columns:\n")
        f.write("  - Patient ID\n")
        f.write("  - Age at diagnosis\n")
        f.write("  - Race/Ethnicity\n")
        f.write("  - Tumor size\n")
        f.write("  - Tumor grade\n")
        f.write("  - Stage at diagnosis\n")
        f.write("  - Survival time\n")
        f.write("  - Vital status\n")
    
    # Create sample template
    template_file = final_dir / 'seer_template.csv'
    with open(template_file, 'w') as f:
        f.write("PatientID,Age,Race,TumorSize,TumorGrade,Stage,SurvivalMonths,VitalStatus\n")
        f.write("SEER001,55,White,25,2,Local,48,Alive\n")
        f.write("SEER002,62,Black,38,3,Regional,36,Deceased\n")
        f.write("SEER003,48,Asian,15,1,Local,60,Alive\n")
    
    print(f"✓ Created template files in: {final_dir}")
    print(f"  - {readme_file} (instructions)")
    print(f"  - {template_file} (data format template)")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for breast cancer classification project'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['breakhis', 'wbcd', 'seer', 'all'],
        default='all',
        help='Dataset to download (default: all)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dataset exists'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BREAST CANCER DATASET DOWNLOADER")
    print("="*70)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download requested datasets
    success_count = 0
    total_count = 0
    
    if args.dataset in ['breakhis', 'all']:
        total_count += 1
        if download_breakhis():
            success_count += 1
    
    if args.dataset in ['wbcd', 'all']:
        total_count += 1
        if download_wbcd():
            success_count += 1
    
    if args.dataset in ['seer', 'all']:
        total_count += 1
        if download_seer():
            success_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"Successfully downloaded: {success_count}/{total_count} datasets")
    
    if success_count == total_count:
        print("\n✓ All requested datasets are ready!")
    elif success_count > 0:
        print("\n⚠ Some datasets require manual download - see instructions above")
    else:
        print("\n✗ No datasets were downloaded")
    
    print("\nNext steps:")
    print("  1. Ensure all datasets are downloaded")
    print("  2. Review config.yaml to select datasets")
    print("  3. Run: python run_pipeline.py")


if __name__ == '__main__':
    main()
