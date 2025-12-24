"""
Similarity computation module for Pathway Explorer.

Handles:
- Jaccard similarity for same-type comparisons (pathway-pathway, TF-TF)
- Overlap Coefficient for cross-type comparisons (TF-pathway)
- Neighbor extraction for edge connections
"""

from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from .config import MIN_JACCARD_EDGE


def compute_hybrid_similarity(
    gene_sets: List[Set[str]],
    entity_types: List[str]
) -> np.ndarray:
    """
    Compute similarity matrix using appropriate metric per comparison type.

    Strategy:
    - Same-type comparisons (TF-TF, Pathway-Pathway, PROGENy-PROGENy): Jaccard
      Jaccard = |A intersection B| / |A union B|

    - TF cross-type comparisons (TF-Pathway, TF-PROGENy): Overlap Coefficient
      Overlap = |A intersection B| / min(|A|, |B|)
      Answers: "What fraction of the smaller set is in the larger?"

    - PROGENy-Pathway comparisons: Jaccard (both are pathway-like enrichments)

    Args:
        gene_sets: List of gene sets (as Python sets)
        entity_types: List of entity types ('TF', 'PROGENy', or 'Pathway')

    Returns:
        Similarity matrix (N x N)
    """
    n = len(gene_sets)
    print(f"Computing hybrid similarity matrix for {n} entities...")
    print(f"  Using Jaccard for same-type, Overlap Coefficient for TF cross-type")

    # Count entity types
    n_tf = sum(1 for t in entity_types if t == 'TF')
    n_pathway = n - n_tf
    print(f"  TFs: {n_tf}, Pathways/PROGENy: {n_pathway}")

    # Precompute set sizes
    set_sizes = np.array([len(gs) for gs in gene_sets])

    # Build binary gene x entity matrix for efficient intersection computation
    all_genes = set()
    for gs in gene_sets:
        all_genes.update(gs)

    gene_to_idx = {g: i for i, g in enumerate(sorted(all_genes))}
    n_genes = len(gene_to_idx)
    print(f"  Total unique genes: {n_genes}")

    # Create sparse binary matrix (genes x entities)
    rows, cols, data = [], [], []
    for entity_idx, gs in enumerate(gene_sets):
        for gene in gs:
            rows.append(gene_to_idx[gene])
            cols.append(entity_idx)
            data.append(1)

    binary_matrix = csr_matrix((data, (rows, cols)), shape=(n_genes, n))

    # Compute intersection matrix: |A intersection B|
    intersection = (binary_matrix.T @ binary_matrix).toarray()

    # Compute union for Jaccard: |A union B| = |A| + |B| - |A intersection B|
    union = set_sizes[:, np.newaxis] + set_sizes[np.newaxis, :] - intersection
    union = np.maximum(union, 1)  # Avoid division by zero

    # Compute min sizes for Overlap Coefficient
    min_sizes = np.minimum(set_sizes[:, np.newaxis], set_sizes[np.newaxis, :])
    min_sizes = np.maximum(min_sizes, 1)  # Avoid division by zero

    # Build similarity matrix with appropriate metric per comparison
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            type_i, type_j = entity_types[i], entity_types[j]

            if type_i == type_j:
                # Same type: use Jaccard
                sim = intersection[i, j] / union[i, j]
            elif 'TF' in (type_i, type_j):
                # TF involved in cross-type comparison: use Overlap Coefficient
                sim = intersection[i, j] / min_sizes[i, j]
            else:
                # Both are pathway-like (GSEA pathway + PROGENy): use Jaccard
                sim = intersection[i, j] / union[i, j]

            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    np.fill_diagonal(sim_matrix, 1.0)

    print(f"  Matrix shape: {sim_matrix.shape}")
    return sim_matrix


def compute_jaccard_matrix(gene_sets: List[Set[str]]) -> np.ndarray:
    """
    Compute pairwise Jaccard similarity matrix for gene sets.

    Note: For mixed TF/pathway data, prefer compute_hybrid_similarity().

    Args:
        gene_sets: List of gene sets (as Python sets)

    Returns:
        Similarity matrix (N x N)
    """
    n = len(gene_sets)
    print(f"Computing Jaccard similarity matrix for {n} pathways...")

    # Build binary gene x pathway matrix
    all_genes = set()
    for gs in gene_sets:
        all_genes.update(gs)

    gene_to_idx = {g: i for i, g in enumerate(sorted(all_genes))}
    n_genes = len(gene_to_idx)
    print(f"  Total unique genes: {n_genes}")

    # Create sparse binary matrix
    rows, cols, data = [], [], []
    for pathway_idx, gs in enumerate(gene_sets):
        for gene in gs:
            rows.append(gene_to_idx[gene])
            cols.append(pathway_idx)
            data.append(1)

    binary_matrix = csr_matrix((data, (rows, cols)), shape=(n_genes, n))

    # Compute Jaccard using set operations
    intersection = binary_matrix.T @ binary_matrix
    intersection = intersection.toarray()

    # Sum of each pathway (number of genes)
    pathway_sizes = np.array(binary_matrix.sum(axis=0)).flatten()

    # Union = |A| + |B| - |A intersection B|
    union = pathway_sizes[:, np.newaxis] + pathway_sizes[np.newaxis, :] - intersection
    union = np.maximum(union, 1)

    jaccard_sim = intersection / union
    np.fill_diagonal(jaccard_sim, 1.0)

    print(f"  Matrix shape: {jaccard_sim.shape}")
    return jaccard_sim


def extract_top_neighbors(
    similarity_matrix: np.ndarray,
    pathway_ids: List[str],
    k: int = 5,
    min_sim: float = MIN_JACCARD_EDGE
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extract k nearest neighbors for each pathway above similarity threshold.

    Args:
        similarity_matrix: Pairwise similarity matrix (N x N)
        pathway_ids: List of pathway IDs matching matrix rows/cols
        k: Number of neighbors to extract per pathway
        min_sim: Minimum similarity threshold for inclusion

    Returns:
        Dict mapping pathway_id -> list of (neighbor_id, similarity) tuples
    """
    neighbors = {}

    for i, pid in enumerate(pathway_ids):
        sims = similarity_matrix[i, :]
        # Get indices of top-k neighbors (excluding self)
        top_indices = np.argsort(sims)[::-1][1:k + 1]

        neighbor_list = []
        for j in top_indices:
            if sims[j] >= min_sim:
                neighbor_list.append((pathway_ids[j], round(float(sims[j]), 3)))

        if neighbor_list:
            neighbors[pid] = neighbor_list

    return neighbors
