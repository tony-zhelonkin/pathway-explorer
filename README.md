# pathway-explorer

Interactive dashboard generator for exploring pathway-level results (GSEA, TF activity, transposable elements) produced by bulk RNA-seq workflows.

## Features
- Consumes standardized GSEA/TF/TE result tables and produces standalone HTML dashboards.
- Supports per-contrast dashboards or a combined all-contrasts view.
- Uses colorblind-safe palettes and configurable thresholds for filtering/labeling.
- Ships with opinionated defaults for embedding, similarity, and data loading steps.

## Installation
```bash
pip install -e .
```

## Usage
- Single contrast:
  ```bash
  pathway-explorer --contrast Sema_WL_vs_Base
  ```
- All contrasts:
  ```bash
  pathway-explorer --all
  ```
- Custom input/output:
  ```bash
  python -m pathway_explorer --data path/to/results.csv --output results/pathway_explorer.html
  ```

Key inputs are defined in `pathway_explorer/config.py`; defaults are in `pathway_explorer/defaults/`.

## Development
- Python >= 3.9.
- Format with `black` and `isort` if available.
- Build artifacts (`__pycache__`, `*.egg-info`) are ignored by default.

## License
This project is licensed under the MIT License (see `LICENSE`).

