"""
Entry point for running pathway_explorer as a module.

Usage:
    python -m pathway_explorer

    # From project root:
    cd /path/to/project
    python -m 02_analysis.pathway_explorer
"""

from .main import main

if __name__ == '__main__':
    main()
