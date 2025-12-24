"""
Pathway Explorer - Interactive Multi-Entity Dashboard Generator.

A modular package for generating interactive pathway exploration dashboards
from GSEA, TF activity, PROGENy, and TE analysis results.

Features:
- Unified visualization of Pathways, TFs, PROGENy, and TEs
- 2D scatter plot with entities positioned by gene overlap (Jaccard-UMAP)
- Color encoding: unified score (diverging blue-white-orange scale)
- Size encoding: -log10(FDR) significance
- Shape encoding: entity type (circle/diamond/square/triangle)
- Edge overlay: shows connected entities (high gene overlap)
- Filtering by database, significance, NES threshold, entity type
- Per-contrast dashboard generation
- Self-contained HTML output

Usage:
    # As a module
    python -m pathway_explorer --contrast Sema_WL_vs_Base

    # Generate all contrasts
    python -m pathway_explorer --all

    # Programmatic
    from pathway_explorer import generate_dashboard
    generate_dashboard(contrast="Sema_WL_vs_Base")
"""

from .config import (
    COLORS,
    DB_COLORS,
    DATA_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    ENTITY_TYPES,
    ENTITY_SHAPES,
)
from .data_loader import (
    clean_pathway_name,
    load_gsea_data,
    standardize_scores,
)
from .embedding import (
    compute_embedding,
    get_best_method,
    UMAP_AVAILABLE,
    SKLEARN_AVAILABLE,
)
from .html_generator import (
    generate_html,
    prepare_pathway_data,
)
from .main import (
    generate_dashboard,
    generate_all_dashboards,
    generate_index_page,
    main,
)
from .similarity import (
    compute_hybrid_similarity,
    compute_jaccard_matrix,
    extract_top_neighbors,
)

__version__ = "2.0.0"
__all__ = [
    # Main entry points
    "generate_dashboard",
    "generate_all_dashboards",
    "generate_index_page",
    "main",
    # Config
    "COLORS",
    "DB_COLORS",
    "DATA_DIR",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "ENTITY_TYPES",
    "ENTITY_SHAPES",
    # Data loading
    "load_gsea_data",
    "clean_pathway_name",
    "standardize_scores",
    # Similarity
    "compute_hybrid_similarity",
    "compute_jaccard_matrix",
    "extract_top_neighbors",
    # Embedding
    "compute_embedding",
    "get_best_method",
    "UMAP_AVAILABLE",
    "SKLEARN_AVAILABLE",
    # HTML generation
    "generate_html",
    "prepare_pathway_data",
]
