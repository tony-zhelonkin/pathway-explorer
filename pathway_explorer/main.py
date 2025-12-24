"""
Main orchestration module for Pathway Explorer.

Coordinates the pipeline:
1. Load unified data (GSEA + TF + PROGENy + TE)
2. Filter by contrast
3. Standardize scores
4. Compute similarity matrix
5. Extract neighbors
6. Compute embedding
7. Generate HTML dashboard
"""

import argparse
from pathlib import Path
from typing import Optional, List

from .config import OUTPUT_DIR, DATA_DIR, ENTITY_SHAPES
from .data_loader import load_gsea_data, standardize_scores
from .embedding import compute_embedding, get_best_method
from .html_generator import generate_html, prepare_pathway_data
from .similarity import compute_hybrid_similarity, extract_top_neighbors


def generate_dashboard(
    data_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    contrast: Optional[str] = None,
    entity_types: Optional[List[str]] = None,
    te_level: str = "family",
) -> Path:
    """
    Generate the interactive pathway explorer dashboard.

    Args:
        data_path: Path to unified data CSV (default: DATA_DIR/master_unified.csv)
        output_path: Output file path (default: OUTPUT_DIR/pathway_explorer_{contrast}.html)
        contrast: Filter to specific contrast (required for per-contrast dashboards)
        entity_types: List of entity types to include (default: all)
        te_level: TE aggregation level: "family" or "class" (default: "family")

    Returns:
        Path to generated HTML file
    """
    print("=" * 70)
    print("PATHWAY EXPLORER GENERATOR")
    print("=" * 70)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    data_file = data_path or (DATA_DIR / "master_unified.csv")
    df = load_gsea_data(data_file)

    if len(df) == 0:
        raise ValueError("No data found in input file!")

    # 2. Filter by contrast if specified
    if contrast:
        print(f"Filtering to contrast: {contrast}")
        if 'contrast' in df.columns:
            df = df[df['contrast'] == contrast].copy()
            if len(df) == 0:
                available = df['contrast'].unique().tolist() if 'contrast' in df.columns else []
                raise ValueError(f"No data found for contrast '{contrast}'. Available: {available}")
        else:
            print("  Warning: 'contrast' column not found, using all data")

    # 3. Filter by entity types if specified
    if entity_types:
        print(f"Filtering to entity types: {entity_types}")
        if 'entity_type' in df.columns:
            df = df[df['entity_type'].isin(entity_types)].copy()
        else:
            print("  Warning: 'entity_type' column not found")

    # 4. Filter TE level if needed
    if te_level and 'database' in df.columns:
        te_filter = f"TE_{te_level.capitalize()}"
        other_te = "TE_Class" if te_level == "family" else "TE_Family"
        # Keep all non-TE entries, plus only the selected TE level
        df = df[~df['database'].str.startswith('TE_') | (df['database'] == te_filter)].copy()
        print(f"  TE level: {te_level} ({te_filter})")

    if len(df) == 0:
        raise ValueError("No pathways found after filtering!")

    print(f"  Total entries: {len(df)}")

    # 5. Standardize scores for unified visualization
    print("Standardizing scores...")
    df = standardize_scores(df)

    # 6. Compute hybrid similarity (Jaccard for same-type, Overlap for cross-type)
    gene_sets = df['genes'].tolist()
    entity_types_list = df['entity_type'].tolist() if 'entity_type' in df.columns else ['Pathway'] * len(df)
    similarity_matrix = compute_hybrid_similarity(gene_sets, entity_types_list)

    # 7. Extract neighbors for edge connections
    pathway_ids = df['pathway_id'].tolist()
    neighbors = extract_top_neighbors(similarity_matrix, pathway_ids, k=5)

    # 8. Compute embedding
    method = get_best_method()
    embedding = compute_embedding(similarity_matrix, method=method)

    # 9. Prepare data for export
    print("Preparing pathway data for export...")
    pathways = prepare_pathway_data(df, embedding, neighbors)

    # Collect entity types for metadata
    entity_type_counts = {}
    if 'entity_type' in df.columns:
        for et in df['entity_type'].unique():
            entity_type_counts[et] = int((df['entity_type'] == et).sum())

    metadata = {
        'total_pathways': len(pathways),
        'databases': sorted(df['database'].unique().tolist()),
        'entity_types': entity_type_counts,
        'entity_shapes': ENTITY_SHAPES,
        'embedding_method': method,
        'contrast': contrast or 'All',
        'te_level': te_level,
    }

    # 10. Generate HTML
    print("Generating HTML dashboard...")
    html = generate_html(pathways, metadata)

    # 11. Write output
    if output_path:
        output_file = output_path
    elif contrast:
        output_file = OUTPUT_DIR / f"pathway_explorer_{contrast}.html"
    else:
        output_file = OUTPUT_DIR / "pathway_explorer.html"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding='utf-8')

    print("=" * 70)
    print(f"Dashboard saved to: {output_file}")
    print(f"   Entries: {len(pathways)}")
    print(f"   Contrast: {contrast or 'All'}")
    print(f"   Entity types: {entity_type_counts}")
    print(f"   Databases: {', '.join(metadata['databases'][:5])}{'...' if len(metadata['databases']) > 5 else ''}")
    print(f"   Embedding: {method.upper()}")
    print("=" * 70)
    print("\nOpen in browser to explore pathway relationships!")

    return output_file


def generate_all_dashboards(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    te_level: str = "family",
) -> List[Path]:
    """
    Generate dashboards for all contrasts in the data.

    Args:
        data_path: Path to unified data CSV
        output_dir: Output directory for HTML files
        te_level: TE aggregation level

    Returns:
        List of paths to generated HTML files
    """
    import pandas as pd

    data_file = data_path or (DATA_DIR / "master_unified.csv")
    out_dir = output_dir or OUTPUT_DIR

    # Load data to get contrast list
    df = pd.read_csv(data_file)
    if 'contrast' not in df.columns:
        raise ValueError("Data file must have 'contrast' column for batch generation")

    contrasts = sorted(df['contrast'].unique())
    print(f"Found {len(contrasts)} contrasts: {contrasts}")

    generated = []
    for contrast in contrasts:
        print(f"\n{'='*70}")
        print(f"Generating dashboard for: {contrast}")
        try:
            output_path = out_dir / f"pathway_explorer_{contrast}.html"
            generate_dashboard(
                data_path=data_file,
                output_path=output_path,
                contrast=contrast,
                te_level=te_level,
            )
            generated.append(output_path)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Generate index page
    generate_index_page(generated, out_dir)

    return generated


def generate_index_page(dashboard_files: List[Path], output_dir: Path) -> Path:
    """
    Generate an index.html page linking to all contrast dashboards.

    Args:
        dashboard_files: List of generated dashboard file paths
        output_dir: Output directory

    Returns:
        Path to generated index.html
    """
    links = []
    for f in sorted(dashboard_files):
        contrast = f.stem.replace('pathway_explorer_', '')
        links.append(f'<li><a href="{f.name}">{contrast}</a></li>')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pathway Explorer - Contrast Index</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 2rem auto;
            padding: 0 1rem;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; border-bottom: 2px solid #2166AC; padding-bottom: 0.5rem; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ margin: 0.5rem 0; }}
        a {{
            display: block;
            padding: 1rem;
            background: white;
            border-radius: 8px;
            text-decoration: none;
            color: #2166AC;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        a:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }}
        .legend {{
            margin-top: 2rem;
            padding: 1rem;
            background: white;
            border-radius: 8px;
        }}
        .legend h3 {{ margin-top: 0; color: #666; }}
        .entity-type {{
            display: inline-flex;
            align-items: center;
            margin-right: 1.5rem;
            font-size: 0.9rem;
        }}
        .shape {{ margin-right: 0.5rem; font-size: 1.2rem; }}
    </style>
</head>
<body>
    <h1>Pathway Explorer</h1>
    <p>Interactive dashboards for each experimental contrast. Click to explore pathway, TF, PROGENy, and TE relationships.</p>
    <ul>
        {''.join(links)}
    </ul>
    <div class="legend">
        <h3>Entity Types</h3>
        <span class="entity-type"><span class="shape">&#9679;</span> Pathway (GSEA)</span>
        <span class="entity-type"><span class="shape">&#9670;</span> TF (CollecTRI)</span>
        <span class="entity-type"><span class="shape">&#9632;</span> PROGENy</span>
        <span class="entity-type"><span class="shape">&#9650;</span> TE Family</span>
    </div>
</body>
</html>
"""

    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding='utf-8')
    print(f"\nIndex page saved to: {index_path}")
    return index_path


def main():
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate interactive pathway explorer dashboards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single dashboard for one contrast
  python -m pathway_explorer --contrast Sema_WL_vs_Base

  # Generate all contrast dashboards
  python -m pathway_explorer --all

  # Custom input/output
  python -m pathway_explorer --data my_data.csv --output my_dashboard.html
        """
    )

    parser.add_argument(
        "--data", "-d",
        type=Path,
        help="Path to unified data CSV (default: 03_results/tables/master_unified.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output HTML file path"
    )
    parser.add_argument(
        "--contrast", "-c",
        type=str,
        help="Filter to specific contrast"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate dashboards for all contrasts"
    )
    parser.add_argument(
        "--te-level",
        type=str,
        choices=["family", "class"],
        default="family",
        help="TE aggregation level (default: family)"
    )
    parser.add_argument(
        "--entity-types", "-e",
        type=str,
        nargs="+",
        choices=["Pathway", "TF", "PROGENy", "TE"],
        help="Filter to specific entity types"
    )

    args = parser.parse_args()

    if args.all:
        generate_all_dashboards(
            data_path=args.data,
            output_dir=args.output.parent if args.output else None,
            te_level=args.te_level,
        )
    else:
        generate_dashboard(
            data_path=args.data,
            output_path=args.output,
            contrast=args.contrast,
            entity_types=args.entity_types,
            te_level=args.te_level,
        )


if __name__ == '__main__':
    main()
