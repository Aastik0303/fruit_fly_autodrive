"""Project-wide paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "backend" / "data"
RAW_PUBLIC_DIR = DATA_DIR / "raw" / "public"
RAW_CODEX_DIR = DATA_DIR / "raw" / "codex"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# FAFB v14.1 electron-microscopy voxel size in nanometres (x, y, z).
VOXEL_SIZE_NM = (4.0, 4.0, 40.0)

# Fill value for missing categorical annotations.
UNKNOWN = "unknown"
