# trading_system/utils/project_paths.py
from pathlib import Path

# Resolve the project root: trading_system/ (the package root)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Common directories
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"  # create if needed
UTILS_DIR = PROJECT_ROOT / "utils"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
VISUALIZATION_DIR = PROJECT_ROOT / "visualization"
ACQUISITION_DIR = PROJECT_ROOT / "data" / "acquisition"