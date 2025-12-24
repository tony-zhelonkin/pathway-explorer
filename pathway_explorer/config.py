"""
Configuration module for Pathway Explorer.

Handles:
- Path configuration
- YAML config loading
- Schema validation
- Visual constants (colors, thresholds)
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Try importing yaml for config loading
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

def get_project_root() -> Path:
    """Get project root directory.

    Attempts to find project root by looking for 03_results/tables directory.
    Falls back to current working directory if not found.
    """
    # Try going up from this file to find project root
    current = Path(__file__).parent.parent.parent.parent
    if (current / "03_results" / "tables").exists():
        return current

    # Try current working directory
    cwd = Path.cwd()
    if (cwd / "03_results" / "tables").exists():
        return cwd

    # Fallback to parent of parent of parent (original behavior)
    return Path(__file__).parent.parent.parent


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "03_results" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "03_results" / "interactive"
CONFIG_FILE = PROJECT_ROOT / "02_analysis" / "config" / "pipeline.yaml"


# ============================================================================
# YAML CONFIG LOADING
# ============================================================================

def load_yaml_config(config_path: Path) -> Optional[dict]:
    """
    Load configuration from YAML file if available.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary or None if unavailable
    """
    if not YAML_AVAILABLE:
        print("Warning: PyYAML not installed. Using hardcoded defaults.")
        return None

    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"Warning: Failed to load config: {e}")
        return None


# Load shared configuration (single source of truth)
PIPELINE_CONFIG = load_yaml_config(CONFIG_FILE)
SCHEMAS = PIPELINE_CONFIG.get('schemas', {}) if PIPELINE_CONFIG else {}


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

def validate_schema(df: pd.DataFrame, schema_name: str, schemas: Optional[dict] = None) -> bool:
    """
    Validate DataFrame against schema definition.

    Args:
        df: DataFrame to validate
        schema_name: Name of schema in schemas dict
        schemas: Dictionary of schema definitions (uses global SCHEMAS if None)

    Returns:
        True if valid, raises ValueError if invalid
    """
    schemas = schemas or SCHEMAS

    if not schemas or schema_name not in schemas:
        print(f"Warning: Schema '{schema_name}' not defined, skipping validation")
        return True

    schema = schemas[schema_name]
    required = schema.get('required_columns', [])

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Schema validation failed for {schema_name}. "
            f"Missing required columns: {missing}"
        )

    print(f"  Schema validation passed: {schema_name} (v{schema.get('version', '?')})")
    return True


# ============================================================================
# FILTER THRESHOLDS
# ============================================================================

# Significance filters for interactive view
MAX_PATHWAYS: Optional[int] = None  # No limit - include all pathways
FDR_THRESHOLD: float = 1.0  # Include all (slider defaults to 0.05 in UI)
MIN_JACCARD_EDGE: float = 0.15  # Minimum Jaccard similarity to draw edge
DEFAULT_FDR_SLIDER: float = 0.05  # Default FDR threshold in UI


# ============================================================================
# VISUAL ENCODING
# ============================================================================

NES_MAX: float = 3.5  # Cap for color scale
PLOTLY_JS_URL: str = "https://cdn.plot.ly/plotly-2.27.0.min.js"

# Colorblind-safe diverging scale (NES/score colors)
COLORS: Dict[str, str] = {
    'negative': '#2166AC',  # Blue (downregulated)
    'neutral': '#F7F7F7',   # White
    'positive': '#B35806',  # Orange (upregulated)
}

# Database colors - Okabe-Ito colorblind-safe palette
# Validated for deuteranopia and protanopia
# Mito databases use analogous violet/purple hues to signal relatedness
DB_COLORS: Dict[str, str] = {
    'Hallmark': '#E69F00',      # Orange
    'KEGG': '#56B4E9',          # Sky Blue
    'Reactome': '#009E73',      # Bluish Green
    'WikiPathways': '#F0E442',  # Yellow
    'GO_BP': '#0072B2',         # Blue
    'GO_MF': '#D55E00',         # Vermillion
    'GO_CC': '#CC79A7',         # Reddish Purple
    'MitoPathways': '#5E4FA2',  # Deep Violet (mito family)
    'MitoXplorer': '#9E9AC8',   # Light Periwinkle (mito family)
    'CollecTRI': '#882255',     # Wine
    'PROGENy': '#44AA99',       # Teal
    'TransportDB': '#DDCC77',   # Sand
    'GATOM': '#117733',         # Forest
    # TE databases - Red spectrum for distinct identification
    'TE_Class': '#8B0000',      # Dark Red
    'TE_Family': '#DC143C',     # Crimson
    'TE_Subfamily': '#FF6347',  # Tomato
    # Additional TF-related
    'TF_Targets': '#AA4499',    # Purple
}

# ============================================================================
# ENTITY TYPE CONFIGURATION
# ============================================================================

# Entity types with their visual encoding
ENTITY_TYPES: Dict[str, Dict[str, str]] = {
    'Pathway': {
        'shape': 'circle',
        'description': 'Gene set enrichment pathways',
    },
    'TF': {
        'shape': 'diamond',
        'description': 'Transcription factor activities',
    },
    'PROGENy': {
        'shape': 'square',
        'description': 'Signaling pathway activities',
    },
    'TE': {
        'shape': 'triangle-up',
        'description': 'Transposable element families',
    },
}

# Plotly marker symbols for entity types
ENTITY_SHAPES: Dict[str, str] = {
    'Pathway': 'circle',
    'TF': 'diamond',
    'PROGENy': 'square',
    'TE': 'triangle-up',
}

# Default entity type if not specified
DEFAULT_ENTITY_TYPE: str = 'Pathway'
