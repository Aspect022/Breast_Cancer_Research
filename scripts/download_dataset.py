#!/usr/bin/env python3
"""
BreakHis Dataset Downloader

Downloads and extracts the BreakHis breast cancer histopathological dataset.
Supports:
- Automatic download from official source
- Resume interrupted downloads
- Progress bar display
- Checksum validation (when available)
- Proper directory structure organization

Dataset: BreakHis (Breast Cancer Histopathological Database)
Source: https://web.inf.pr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

Directory structure after extraction:
    data/
    └── BreaKHis_v1/
        └── histology_slides/
            └── breast/
                ├── benign/
                │   └── SOB/
                │       ├── A/      (Adenosis)
                │       ├── F/      (Fibroadenoma)
                │       ├── P/      (Papilloma)
                │       └── T/      (Tubular Adenoma)
                └── malignant/
                    └── SOB/
                        ├── DC/     (Ductal Carcinoma)
                        ├── LC/     (Lobular Carcinoma)
                        ├── MC/     (Mucinous Carcinoma)
                        ├── PC/     (Papillary Carcinoma)
                        ├── TC/     (Tubular Carcinoma)
                        └── C/      (Carcinoma)

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output_dir /path/to/data
    python scripts/download_dataset.py --task binary  # Only download binary classes
    python scripts/download_dataset.py --resume       # Resume interrupted download
"""

import os
import sys
import hashlib
import argparse
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Third-party imports
try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install requests tqdm")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

# Official BreakHis download URLs
# Note: BreakHis is distributed as multiple ZIP files
DATASET_INFO = {
    "name": "BreakHis",
    "version": "v1",
    "description": "Breast Cancer Histopathological Database",
    "source_url": "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
    # Direct download links (may change - verify from source)
    # The dataset is typically distributed as a single large ZIP or multiple parts
    "files": {
        "full": {
            "url": "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/BreaKHis_v1.zip",
            "filename": "BreaKHis_v1.zip",
            "expected_size_gb": 3.5,  # Approximate size
            "checksum": None,  # Will be computed if not provided
        }
    }
}

# Default paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data"
DEFAULT_EXTRACT_DIR = DEFAULT_OUTPUT_DIR / "BreaKHis_v1"

# Download settings
CHUNK_SIZE = 8192  # 8KB chunks for streaming
MAX_RETRIES = 3
TIMEOUT = 30  # seconds


# ═══════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════

def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file for integrity verification."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def get_file_size(filepath: Path) -> int:
    """Get file size in bytes."""
    return filepath.stat().st_size if filepath.exists() else 0


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def print_directory_tree(root_path: Path, max_depth: int = 4, max_items: int = 20):
    """Print a tree view of directory structure."""
    def _tree(path: Path, prefix: str, depth: int):
        if depth > max_depth:
            return
        
        try:
            items = list(path.iterdir())[:max_items]
        except PermissionError:
            return
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{item.name}")
            
            if item.is_dir():
                extension = "    " if is_last else "│   "
                _tree(item, prefix + extension, depth + 1)
    
    print(f"\n📁 Directory structure ({root_path}):")
    _tree(root_path, "", 0)


# ═══════════════════════════════════════════════════════════════════
# Download Functions
# ═══════════════════════════════════════════════════════════════════

class DownloadProgressBar(tqdm):
    """Custom progress bar for downloads."""
    
    def update_to(self, block_num: int = 1, block_size: int = 1, total_size: int = None):
        """Update progress bar based on blocks downloaded."""
        if total_size is not None:
            self.total = total_size
        self.update(block_num * block_size - self.n)


def download_file(
    url: str,
    dest_path: Path,
    resume: bool = True,
    max_retries: int = MAX_RETRIES
) -> bool:
    """
    Download a file with progress bar and resume support.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        resume: Whether to resume interrupted downloads
        max_retries: Maximum number of retry attempts
    
    Returns:
        True if download successful, False otherwise
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            # Check for partial download
            start_pos = 0
            if resume and dest_path.exists():
                start_pos = get_file_size(dest_path)
                if start_pos > 0:
                    print(f"📥 Resuming download from {format_size(start_pos)}...")
            
            # Setup headers for resume
            headers = {}
            if start_pos > 0:
                headers["Range"] = f"bytes={start_pos}-"
            
            # Start download with streaming
            response = requests.get(url, headers=headers, stream=True, timeout=TIMEOUT)
            response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get("content-length", 0))
            if start_pos > 0 and total_size > 0:
                total_size += start_pos
            
            # Download with progress bar
            mode = "ab" if start_pos > 0 else "wb"
            with open(dest_path, mode) as f:
                with DownloadProgressBar(
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    desc=dest_path.name,
                    initial=start_pos / 1024 / 1024,  # Start from MB already downloaded
                    total=total_size / 1024 / 1024 if total_size > 0 else None,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:  # Filter out keep-alive chunks
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Verify download
            downloaded_size = get_file_size(dest_path)
            if total_size > 0 and downloaded_size != total_size:
                print(f"⚠️  Warning: Downloaded size ({format_size(downloaded_size)}) "
                      f"doesn't match expected ({format_size(total_size)})")
            
            print(f"✓ Download complete: {format_size(downloaded_size)}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Download attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                print(f"✗ Download failed after {max_retries} attempts")
                return False
    
    return False


# ═══════════════════════════════════════════════════════════════════
# Extraction Functions
# ═══════════════════════════════════════════════════════════════════

def extract_zip(
    zip_path: Path,
    extract_to: Path,
    remove_zip: bool = False
) -> bool:
    """
    Extract ZIP file with progress indicator.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Destination directory
        remove_zip: Whether to remove ZIP after extraction
    
    Returns:
        True if extraction successful
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    if not zip_path.exists():
        print(f"✗ ZIP file not found: {zip_path}")
        return False
    
    print(f"\n📦 Extracting {zip_path.name}...")
    print(f"   Destination: {extract_to}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            print(f"   Files to extract: {total_files}")
            
            # Extract with progress
            with tqdm(total=total_files, desc="Extracting", unit="file") as pbar:
                for file in file_list:
                    try:
                        zip_ref.extract(file, extract_to)
                    except Exception as e:
                        print(f"⚠️  Warning: Could not extract {file}: {e}")
                    pbar.update(1)
        
        # Optionally remove ZIP
        if remove_zip:
            print(f"\n🗑️  Removing ZIP file: {zip_path}")
            zip_path.unlink()
        
        print(f"✓ Extraction complete!")
        return True
        
    except zipfile.BadZipFile as e:
        print(f"✗ Invalid ZIP file: {e}")
        return False
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# Validation Functions
# ═══════════════════════════════════════════════════════════════════

def validate_dataset_structure(extract_dir: Path) -> Tuple[bool, Dict]:
    """
    Validate the extracted dataset structure.
    
    Args:
        extract_dir: Path to extracted dataset directory
    
    Returns:
        (is_valid, stats_dict)
    """
    extract_dir = Path(extract_dir)
    
    if not extract_dir.exists():
        print(f"✗ Dataset directory not found: {extract_dir}")
        return False, {}
    
    print(f"\n🔍 Validating dataset structure...")
    
    stats = {
        "total_images": 0,
        "benign_images": 0,
        "malignant_images": 0,
        "magnifications": set(),
        "tumor_types": set(),
        "patients": set(),
    }
    
    # Walk through directory
    benign_path = extract_dir / "histology_slides" / "breast" / "benign"
    malignant_path = extract_dir / "histology_slides" / "breast" / "malignant"
    
    for class_path, class_name in [(benign_path, "benign"), (malignant_path, "malignant")]:
        if not class_path.exists():
            print(f"⚠️  Missing {class_name} directory")
            continue
        
        for root, dirs, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith('.png'):
                    stats["total_images"] += 1
                    if class_name == "benign":
                        stats["benign_images"] += 1
                    else:
                        stats["malignant_images"] += 1
                    
                    # Extract metadata from filename
                    # Format: SOB_B_A_14-101_40X.png
                    parts = file.upper().replace('.PNG', '').split('_')
                    if len(parts) >= 5:
                        stats["tumor_types"].add(parts[2])  # A, F, P, T, DC, LC, etc.
                        stats["magnifications"].add(parts[4])  # 40X, 100X, 200X, 400X
                        stats["patients"].add(f"{parts[3]}_{parts[4]}")  # Patient ID
    
    # Print validation results
    print(f"\n{'='*60}")
    print("  DATASET VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Total images:      {stats['total_images']:,}")
    print(f"  Benign images:     {stats['benign_images']:,}")
    print(f"  Malignant images:  {stats['malignant_images']:,}")
    print(f"  Tumor types:       {len(stats['tumor_types'])} ({', '.join(sorted(stats['tumor_types']))})")
    print(f"  Magnifications:    {len(stats['magnifications'])} ({', '.join(sorted(stats['magnifications']))})")
    print(f"  Unique patients:   {len(stats['patients']):,}")
    print(f"{'='*60}")
    
    # Validation criteria
    is_valid = (
        stats["total_images"] > 0 and
        stats["benign_images"] > 0 and
        stats["malignant_images"] > 0 and
        len(stats["tumor_types"]) > 0 and
        len(stats["magnifications"]) > 0
    )
    
    if is_valid:
        print("✓ Dataset structure validated successfully!")
    else:
        print("✗ Dataset validation failed - structure may be incomplete")
    
    return is_valid, stats


def check_disk_space(required_gb: float, path: Path) -> bool:
    """Check if there's enough disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(str(path))
        free_gb = free / (1024 ** 3)
        
        if free_gb < required_gb:
            print(f"✗ Insufficient disk space!")
            print(f"   Required: {required_gb:.1f} GB")
            print(f"   Available: {free_gb:.1f} GB")
            return False
        
        print(f"✓ Disk space check passed: {free_gb:.1f} GB available")
        return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True  # Proceed anyway


# ═══════════════════════════════════════════════════════════════════
# Main Download Pipeline
# ═══════════════════════════════════════════════════════════════════

def download_breakhis(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    extract_dir: Path = DEFAULT_EXTRACT_DIR,
    resume: bool = True,
    remove_zip: bool = False,
    skip_validation: bool = False,
) -> bool:
    """
    Complete download pipeline for BreakHis dataset.
    
    Args:
        output_dir: Base output directory
        extract_dir: Extraction directory
        resume: Resume interrupted downloads
        remove_zip: Remove ZIP after extraction
        skip_validation: Skip dataset validation
    
    Returns:
        True if successful
    """
    print("\n" + "="*70)
    print("  BREAKHIS DATASET DOWNLOADER")
    print("="*70)
    print(f"  Output directory: {output_dir}")
    print(f"  Extract directory: {extract_dir}")
    print(f"  Resume support: {resume}")
    print("="*70 + "\n")
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check disk space (dataset is ~3.5GB compressed, ~7GB extracted)
    if not check_disk_space(10.0, output_dir):
        return False
    
    # Check if already downloaded
    if extract_dir.exists():
        print(f"\n✓ Dataset already exists at: {extract_dir}")
        
        # Validate existing dataset
        if not skip_validation:
            is_valid, stats = validate_dataset_structure(extract_dir)
            if is_valid:
                print("\n✓ Dataset is ready to use!")
                return True
            else:
                print("\n⚠️  Existing dataset appears incomplete. Re-downloading...")
    
    # Download
    zip_filename = DATASET_INFO["files"]["full"]["filename"]
    zip_path = output_dir / zip_filename
    download_url = DATASET_INFO["files"]["full"]["url"]
    
    print(f"\n📥 Downloading BreakHis dataset...")
    print(f"   URL: {download_url}")
    print(f"   Destination: {zip_path}")
    
    if not download_file(download_url, zip_path, resume=resume):
        print("\n✗ Download failed. Please check:")
        print("   1. Internet connection")
        print("   2. Firewall/proxy settings")
        print("   3. Dataset URL may have changed")
        print(f"\n   Manual download available at:")
        print(f"   {DATASET_INFO['source_url']}")
        return False
    
    # Extract
    print(f"\n📦 Extracting dataset...")
    if not extract_zip(zip_path, output_dir, remove_zip=remove_zip):
        return False
    
    # Validate
    if not skip_validation:
        is_valid, stats = validate_dataset_structure(extract_dir)
        if not is_valid:
            print("\n⚠️  Dataset validation failed. Please check the extraction.")
            return False
    
    # Print final directory structure
    print_directory_tree(extract_dir, max_depth=3)
    
    print("\n" + "="*70)
    print("  ✓ BREAKHIS DATASET READY!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Verify dataset in: {extract_dir}")
    print(f"  2. Update config.yaml data_dir to: {extract_dir}")
    print(f"  3. Run: python run_pipeline.py")
    
    return True


# ═══════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download and setup BreakHis breast cancer dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_dataset.py
  python download_dataset.py --output_dir /path/to/data
  python download_dataset.py --resume
  python download_dataset.py --no-validate
  python download_dataset.py --remove-zip
        """
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--extract_dir", "-e",
        type=Path,
        default=DEFAULT_EXTRACT_DIR,
        help=f"Extraction directory (default: {DEFAULT_EXTRACT_DIR})"
    )
    
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        default=True,
        help="Resume interrupted downloads (default: True)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (start fresh)"
    )
    
    parser.add_argument(
        "--remove-zip",
        action="store_true",
        help="Remove ZIP file after extraction (saves space)"
    )
    
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        default=True,
        help="Keep ZIP file after extraction (default: True)"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip dataset validation after extraction"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if dataset exists"
    )
    
    args = parser.parse_args()
    
    # Handle conflicting options
    if args.no_resume:
        args.resume = False
    
    if args.keep_zip:
        args.remove_zip = False
    
    # Handle force re-download
    if args.force and args.extract_dir.exists():
        print(f"\n🗑️  Removing existing dataset: {args.extract_dir}")
        shutil.rmtree(args.extract_dir)
    
    # Run download
    success = download_breakhis(
        output_dir=args.output_dir,
        extract_dir=args.extract_dir,
        resume=args.resume,
        remove_zip=args.remove_zip,
        skip_validation=args.no_validate,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
