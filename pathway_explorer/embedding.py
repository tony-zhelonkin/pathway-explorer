"""
Dimensionality reduction module for Pathway Explorer.

Handles:
- UMAP embedding (preferred)
- t-SNE embedding (fallback)
- PCA embedding (fallback)
- Random projection (last resort)
"""

import warnings
from typing import Literal

import numpy as np

# Suppress warnings during import checks
warnings.filterwarnings('ignore')

# Check UMAP availability
try:
    import umap.umap_ as umap_module
    UMAP_AVAILABLE = True
except ImportError:
    try:
        from umap import UMAP as UMAP_CLASS
        UMAP_AVAILABLE = True
        umap_module = None
    except ImportError:
        UMAP_AVAILABLE = False
        umap_module = None
        UMAP_CLASS = None

# Check sklearn availability
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None
    TSNE = None


EmbeddingMethod = Literal['umap', 'tsne', 'pca', 'random']


def get_best_method() -> EmbeddingMethod:
    """
    Get the best available embedding method.

    Returns:
        Method name: 'umap' > 'pca' > 'random'
    """
    if UMAP_AVAILABLE:
        return 'umap'
    elif SKLEARN_AVAILABLE:
        return 'pca'
    else:
        return 'random'


def compute_embedding(
    similarity_matrix: np.ndarray,
    method: EmbeddingMethod = 'umap',
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute 2D embedding from similarity matrix.

    Args:
        similarity_matrix: Pairwise similarity matrix (N x N)
        method: Embedding method ('umap', 'tsne', 'pca', 'random')
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed for reproducibility

    Returns:
        2D coordinates (N x 2), normalized to [0, 1] range
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)

    print(f"Computing {method.upper()} embedding...")

    if method == 'umap' and UMAP_AVAILABLE:
        embedding = _compute_umap(distance_matrix, n_neighbors, min_dist, random_state)
    elif method == 'tsne' and SKLEARN_AVAILABLE:
        embedding = _compute_tsne(distance_matrix, random_state)
    elif method == 'pca' and SKLEARN_AVAILABLE:
        embedding = _compute_pca(similarity_matrix, random_state)
    else:
        embedding = _compute_random(len(similarity_matrix), random_state)

    # Normalize to [0, 1] range
    embedding = _normalize_embedding(embedding)

    print(f"  Embedding shape: {embedding.shape}")
    return embedding


def _compute_umap(
    distance_matrix: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int
) -> np.ndarray:
    """Compute UMAP embedding from precomputed distances."""
    if umap_module is not None:
        reducer = umap_module.UMAP(
            metric='precomputed',
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state
        )
    else:
        reducer = UMAP_CLASS(
            metric='precomputed',
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state
        )
    return reducer.fit_transform(distance_matrix)


def _compute_tsne(distance_matrix: np.ndarray, random_state: int) -> np.ndarray:
    """Compute t-SNE embedding from precomputed distances."""
    n_samples = len(distance_matrix)
    perplexity = min(30, n_samples - 1)

    tsne = TSNE(
        n_components=2,
        metric='precomputed',
        perplexity=perplexity,
        random_state=random_state
    )
    return tsne.fit_transform(distance_matrix)


def _compute_pca(similarity_matrix: np.ndarray, random_state: int) -> np.ndarray:
    """Compute PCA embedding from similarity matrix."""
    pca = PCA(n_components=2, random_state=random_state)
    embedding = pca.fit_transform(similarity_matrix)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    return embedding


def _compute_random(n_samples: int, random_state: int) -> np.ndarray:
    """Generate random projection as fallback."""
    print("  Warning: Using random projection (no sklearn available)")
    np.random.seed(random_state)
    return np.random.randn(n_samples, 2)


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding to [0, 1] range."""
    min_vals = embedding.min(axis=0)
    max_vals = embedding.max(axis=0)
    range_vals = max_vals - min_vals + 1e-10  # Avoid division by zero
    return (embedding - min_vals) / range_vals
