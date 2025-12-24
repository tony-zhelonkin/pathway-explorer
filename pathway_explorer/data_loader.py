"""
Data loading module for Pathway Explorer.

Handles:
- Unified table loading (master_unified.csv)
- Legacy GSEA results loading from separate master tables
- TF and PROGENy activity integration
- Pathway name cleaning
- Score standardization for unified visualization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from .config import (
    DATA_DIR,
    FDR_THRESHOLD,
    MAX_PATHWAYS,
    SCHEMAS,
    validate_schema,
    DEFAULT_ENTITY_TYPE,
)


def load_gsea_data(
    data_path: Optional[Union[str, Path]] = None,
    fdr_threshold: float = FDR_THRESHOLD,
) -> pd.DataFrame:
    """
    Load and prepare data from unified or separate master tables.

    Args:
        data_path: Path to data file. If None, tries master_unified.csv first,
                   then falls back to loading separate files.
        fdr_threshold: FDR threshold for filtering (default: include all)

    Returns:
        DataFrame with columns: pathway_id, pathway_name, database, entity_type,
        nes, padj, pvalue, set_size, leading_edge_size, core_enrichment, genes,
        display_name, signed_sig
    """
    # Determine data source
    if data_path:
        data_file = Path(data_path)
    elif (DATA_DIR / "master_unified.csv").exists():
        data_file = DATA_DIR / "master_unified.csv"
    elif (DATA_DIR / "master_gsea_table.csv").exists():
        data_file = DATA_DIR / "master_gsea_table.csv"
    elif (DATA_DIR / "master_gsea.csv").exists():
        data_file = DATA_DIR / "master_gsea.csv"
    else:
        raise FileNotFoundError(
            f"No data file found. Looked for master_unified.csv, "
            f"master_gsea_table.csv, master_gsea.csv in {DATA_DIR}"
        )

    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"  Total entries: {len(df)}")

    # Check if this is unified format (has entity_type column)
    is_unified = 'entity_type' in df.columns

    if is_unified:
        print("  Format: Unified table (entity_type column present)")
        # Use signed_sig if already computed
        if 'signed_sig' not in df.columns:
            df = standardize_scores(df)
    else:
        print("  Format: Legacy (separate files)")
        # Validate schema for legacy format
        if SCHEMAS:
            validate_schema(df, 'master_gsea_table', SCHEMAS)

        # Load TF activities if available
        df = _load_tf_activities(df)

        # Load PROGENy activities if available
        df = _load_progeny_activities(df)

        # Reclassify Mitochondria into specific sources
        df = _reclassify_mito_databases(df)

        # Add entity_type column
        df = _add_entity_types(df)

        # Standardize scores
        df = standardize_scores(df)

    # Filter by FDR if requested (default includes all)
    if fdr_threshold < 1.0:
        df_sig = df[df['padj'] <= fdr_threshold].copy()
        print(f"  After FDR filter ({fdr_threshold}): {len(df_sig)}")
    else:
        df_sig = df.copy()

    # Optionally limit if MAX_PATHWAYS is set
    if MAX_PATHWAYS is not None and len(df_sig) > MAX_PATHWAYS:
        df_sig = df_sig.nsmallest(MAX_PATHWAYS, 'padj')
        print(f"  Limited to top {MAX_PATHWAYS} by significance")

    # Parse gene sets from core_enrichment column
    df_sig['genes'] = df_sig['core_enrichment'].apply(
        lambda x: set(str(x).split('/')) if pd.notna(x) and str(x) not in ['nan', 'NA', ''] else set()
    )

    # Clean up pathway names for display
    df_sig['display_name'] = df_sig['pathway_name'].apply(clean_pathway_name)

    # Ensure entity_type column exists
    if 'entity_type' not in df_sig.columns:
        df_sig['entity_type'] = DEFAULT_ENTITY_TYPE

    print(f"  Final dataset: {len(df_sig)} entries")
    print(f"  Entity types: {df_sig['entity_type'].value_counts().to_dict()}")
    print(f"  Databases: {sorted(df_sig['database'].unique().tolist())}")

    return df_sig


def _load_tf_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Load and concatenate TF activities if available."""
    tf_file = DATA_DIR / "master_tf_activities.csv"

    if not tf_file.exists():
        return df

    print(f"Loading TF activities from {tf_file}...")
    df_tf = pd.read_csv(tf_file)
    print(f"  Total TFs: {len(df_tf)}")

    if SCHEMAS:
        validate_schema(df_tf, 'master_tf_activities', SCHEMAS)

    df = pd.concat([df, df_tf], ignore_index=True)
    print(f"  Combined Total: {len(df)}")

    return df


def _load_progeny_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Load and concatenate PROGENy activities if available."""
    progeny_file = DATA_DIR / "master_progeny_activities.csv"

    if not progeny_file.exists():
        return df

    print(f"Loading PROGENy activities from {progeny_file}...")
    df_progeny = pd.read_csv(progeny_file)
    print(f"  Total PROGENy pathways: {len(df_progeny)}")

    if SCHEMAS:
        validate_schema(df_progeny, 'master_tf_activities', SCHEMAS)

    df = pd.concat([df, df_progeny], ignore_index=True)
    print(f"  Combined Total: {len(df)}")

    return df


def _reclassify_mito_databases(df: pd.DataFrame) -> pd.DataFrame:
    """Reclassify generic 'Mitochondria' database into specific sources."""

    def reclassify_mito(row):
        if row['database'] == 'Mitochondria':
            pid = str(row['pathway_id'])
            if pid.startswith('MITOPATHWAYS_'):
                return 'MitoPathways'
            elif pid.startswith('MITOXPLORER_'):
                return 'MitoXplorer'
        return row['database']

    if 'Mitochondria' in df['database'].values:
        df['database'] = df.apply(reclassify_mito, axis=1)
        print(f"  Reclassified Mitochondria -> MitoPathways/MitoXplorer")

    return df


def _add_entity_types(df: pd.DataFrame) -> pd.DataFrame:
    """Add entity_type column based on database."""

    def get_entity_type(db: str) -> str:
        if db == 'CollecTRI':
            return 'TF'
        elif db == 'PROGENy':
            return 'PROGENy'
        elif str(db).startswith('TE_'):
            return 'TE'
        else:
            return 'Pathway'

    df['entity_type'] = df['database'].apply(get_entity_type)

    counts = df['entity_type'].value_counts()
    print(f"  Entity types: {counts.to_dict()}")

    return df


def clean_pathway_name(name: str, max_length: int = 60) -> str:
    """
    Clean pathway name for display.

    Args:
        name: Raw pathway name
        max_length: Maximum display length

    Returns:
        Cleaned, title-cased, truncated name
    """
    if pd.isna(name):
        return "Unknown"

    # Title case, replace underscores
    cleaned = str(name).replace('_', ' ').title()

    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length - 3] + "..."

    return cleaned


def standardize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unified score for visualization across TFs, pathways, and TEs.

    Computes: signed_sig = -log10(padj) * sign(nes)

    This transformation:
    - Makes TF activity scores, GSEA NES, and TE effects comparable
    - Preserves direction (positive = upregulated, negative = downregulated)
    - Emphasizes statistical significance rather than effect size alone

    Args:
        df: DataFrame with 'padj' and 'nes' (or 'NES') columns

    Returns:
        DataFrame with added 'signed_sig' column (capped at +/-50)
    """
    df = df.copy()

    # Handle different column name conventions
    nes_col = 'NES' if 'NES' in df.columns else 'nes'
    padj_col = 'padj' if 'padj' in df.columns else 'adj.P.Val'

    if nes_col not in df.columns:
        raise ValueError(f"NES column not found. Available: {df.columns.tolist()}")
    if padj_col not in df.columns:
        raise ValueError(f"padj column not found. Available: {df.columns.tolist()}")

    # Normalize column names
    if 'NES' in df.columns and 'nes' not in df.columns:
        df['nes'] = df['NES']
    if 'adj.P.Val' in df.columns and 'padj' not in df.columns:
        df['padj'] = df['adj.P.Val']

    # Compute signed significance: -log10(padj) * sign(nes)
    # Clip padj to avoid log(0) = -inf
    padj_clipped = df['padj'].clip(lower=1e-50)
    neg_log_padj = -np.log10(padj_clipped)

    # Apply sign based on NES direction
    df['signed_sig'] = neg_log_padj * np.sign(df['nes'])

    # Cap extreme values to prevent outliers dominating color scale
    df['signed_sig'] = df['signed_sig'].clip(lower=-50, upper=50)

    print(f"  Score standardization: signed_sig range "
          f"[{df['signed_sig'].min():.1f}, {df['signed_sig'].max():.1f}]")

    return df
