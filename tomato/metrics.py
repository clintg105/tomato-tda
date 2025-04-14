from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
from sklearn.metrics.pairwise import (
    cosine_distances,
    euclidean_distances,
    manhattan_distances,
    pairwise_distances
)

import ot

from scipy.spatial import distance
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph


def pdist2(X, Y, metric="cosine", p=2):
    """
    Fast pairwise distance between rows of X and rows of Y (n_points, dim).
    Recognized strings: 'cosine', 'minkowski'.
      - Minkowski with p=1 => manhattan_distances
      - Minkowski with p=2 => euclidean_distances
      - Minkowski with other p => pairwise_distances(..., metric='minkowski')
    If metric is a callable, fallback to a Python loop (slower).
    """
    if isinstance(metric, str):
        m = metric.lower()
        if m == "cosine":
            return cosine_distances(X, Y)
        elif m == "minkowski":
            if p == 1:
                return manhattan_distances(X, Y)
            elif p == 2:
                return euclidean_distances(X, Y)
            else:
                return pairwise_distances(X, Y, metric="minkowski", p=p)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    elif callable(metric):
        n, m = X.shape[0], Y.shape[0]
        D = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                D[i, j] = metric(X[i], Y[j])
        return D
    else:
        raise ValueError("metric must be a string or callable.")

def wasserstein_distance(X, Y, ground_metric="cosine", p=2):
    """
    1-Wasserstein distance (Earth Mover's Distance) 
    for uniform distributions over columns of X and Y.
    X is (n_points, dim), Y is (n_points, dim).
    """
    # Cost matrix
    C = pdist2(X, Y, ground_metric, p=p)
    # Uniform weights
    n, m = X.shape[0], Y.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m
    return ot.emd2(a, b, C)

def wasserstein_distances(df_enc, ground_metric="cosine", p=2):
    dim_cols = [c for c in df_enc.columns if c.startswith('dim_')]
    group_dstrbs = [group[dim_cols].values for _, group in df_enc.groupby('review_id')]
    n = len(group_dstrbs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_ij = wasserstein_distance(group_dstrbs[i], group_dstrbs[j], ground_metric, p=p)
            D[i, j] = dist_ij
            D[j, i] = dist_ij
    return D
def wasserstein_distances_parallel(df_enc, ground_metric="cosine", p=2, nCores=-1):
    """ This is actually slower... Memory to workers? """
    dim_cols = [c for c in df_enc.columns if c.startswith("dim_")]
    group_dstrbs = [group[dim_cols].values for _, group in df_enc.groupby("review_id")]
    n = len(group_dstrbs)

    def compute_ij(i, j):
        d = wasserstein_distance(group_dstrbs[i], group_dstrbs[j], ground_metric, p=p)
        return (i, j, d)

    tasks = [(i, j) for i in range(n) for j in range(i+1, n)]
    results = Parallel(n_jobs=nCores)(
        delayed(compute_ij)(i, j) for i, j in tqdm(tasks, desc="Wasserstein Dists")
    )

    D = np.zeros((n, n))
    for i, j, d in results:
        D[i, j] = d
        D[j, i] = d
    return D