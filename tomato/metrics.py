from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
from sklearn.metrics.pairwise import (
    cosine_distances,
    euclidean_distances,
    manhattan_distances,
    pairwise_distances,
)

import ot

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA


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
            elif p == np.inf:
                return pairwise_distances(X, Y, metric="chebyshev")
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
    dim_cols = [c for c in df_enc.columns if c.startswith("dim_")]
    group_dstrbs = [group[dim_cols].values for _, group in df_enc.groupby("review_id")]
    n = len(group_dstrbs)
    D = np.zeros((n, n))
    for i in range(n):

        for j in range(i + 1, n):
            dist_ij = wasserstein_distance(
                group_dstrbs[i], group_dstrbs[j], ground_metric, p=p
            )
            D[i, j] = dist_ij
            D[j, i] = dist_ij
    return D


def wasserstein_distances_parallel(df_enc, ground_metric="cosine", p=2, nCores=-1):
    """This is actually slower... Memory to workers?"""
    dim_cols = [c for c in df_enc.columns if c.startswith("dim_")]
    group_dstrbs = [group[dim_cols].values for _, group in df_enc.groupby("review_id")]
    n = len(group_dstrbs)

    def compute_ij(gi, gj, i, j):
        d = wasserstein_distance(gi, gj, ground_metric, p=p)
        return (i, j, d)

    tasks = [(i, j) for i in range(n) for j in range(i + 1, n)]
    results = Parallel(n_jobs=nCores)(
        delayed(compute_ij)(group_dstrbs[i], group_dstrbs[j], i, j)
        for i, j in tqdm(tasks, desc="Wasserstein Dists")
    )

    D = np.zeros((n, n))
    for i, j, d in results:
        D[i, j] = d
        D[j, i] = d
    return D


def wasserstein_distance_sinkhorn(X, Y, ground_metric="cosine", p=2, reg=0.1):
    """
    Approximates the Wasserstein distance using Sinkhorn algorithm.
    Regularization parameter controls accuracy vs. speed tradeoff.
    Higher reg = faster but less accurate approximation.
    """
    C = pdist2(X, Y, ground_metric, p=p)
    n, m = X.shape[0], Y.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m
    return ot.sinkhorn2(a, b, C, reg=reg)


def wasserstein_distances_sinkhorn(df_enc, ground_metric="cosine", p=2, reg=0.1):
    dim_cols = [c for c in df_enc.columns if c.startswith("dim_")]
    group_dstrbs = [group[dim_cols].values for _, group in df_enc.groupby("review_id")]
    n = len(group_dstrbs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = wasserstein_distance_sinkhorn(
                group_dstrbs[i], group_dstrbs[j], ground_metric, p=p, reg=reg
            )
            D[i, j] = dist_ij
            D[j, i] = dist_ij
    return D


def wasserstein_distances_sinkhorn_parallel(
    df_enc, ground_metric="cosine", p=2, reg=0.1, nCores=-1, batch_size=10
):
    dim_cols = [c for c in df_enc.columns if c.startswith("dim_")]
    group_dstrbs = [group[dim_cols].values for _, group in df_enc.groupby("review_id")]
    n = len(group_dstrbs)

    def compute_ij(gi, gj, i, j):
        d = wasserstein_distance_sinkhorn(gi, gj, ground_metric, p=p, reg=reg)
        return (i, j, d)

    tasks = [(i, j) for i in range(n) for j in range(i + 1, n)]

    D = np.zeros((n, n))
    for i in tqdm(
        range(0, len(tasks), batch_size), desc="Sinkhorn Wasserstein Batches"
    ):
        batch_tasks = tasks[i : i + batch_size]
        batch_results = Parallel(n_jobs=nCores)(
            delayed(compute_ij)(
                group_dstrbs[task[0]], group_dstrbs[task[1]], task[0], task[1]
            )
            for task in batch_tasks
        )

        for i, j, d in batch_results:
            D[i, j] = d
            D[j, i] = d

    return D


def sliced_wasserstein_distance(X, Y, ground_metric="cosine", p=2, n_projections=50):
    """
    Sliced Wasserstein distance projects high-dimensional distributions to 1D
    and computes the average Wasserstein distance along random projections.
    This is much faster than exact EMD while preserving similarity structure.
    """
    if ground_metric == "cosine":
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
        X_use, Y_use = X_norm, Y_norm
    else:
        X_use, Y_use = X, Y

    dim = X_use.shape[1]
    n_x, n_y = X_use.shape[0], Y_use.shape[0]

    projections = np.random.normal(size=(n_projections, dim))
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)

    X_projs = np.dot(X_use, projections.T)
    Y_projs = np.dot(Y_use, projections.T)

    distances = []
    for j in range(n_projections):
        X_proj = X_projs[:, j].reshape(-1, 1)
        Y_proj = Y_projs[:, j].reshape(-1, 1)

        X_proj_sorted = np.sort(X_proj.flatten())
        Y_proj_sorted = np.sort(Y_proj.flatten())

        a = np.ones(n_x) / n_x
        b = np.ones(n_y) / n_y

        if n_x == n_y:
            dist = np.mean(np.abs(X_proj_sorted - Y_proj_sorted))
        else:
            X_quantiles = np.interp(
                np.linspace(0, 1, n_y), np.linspace(0, 1, n_x), X_proj_sorted
            )
            dist = np.mean(np.abs(X_quantiles - Y_proj_sorted))

        distances.append(dist)

    return np.mean(distances)


def sliced_wasserstein_distances(df_enc, ground_metric="cosine", p=2, n_projections=50):
    dim_cols = [c for c in df_enc.columns if c.startswith("dim_")]
    group_dstrbs = [group[dim_cols].values for _, group in df_enc.groupby("review_id")]
    n = len(group_dstrbs)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = sliced_wasserstein_distance(
                group_dstrbs[i],
                group_dstrbs[j],
                ground_metric,
                p=p,
                n_projections=n_projections,
            )
            D[i, j] = dist_ij
            D[j, i] = dist_ij
    return D


def sliced_wasserstein_distances_parallel(
    df_enc, ground_metric="cosine", p=2, n_projections=50, nCores=-1, batch_size=10
):
    dim_cols = [c for c in df_enc.columns if c.startswith("dim_")]
    group_dstrbs = [group[dim_cols].values for _, group in df_enc.groupby("review_id")]
    n = len(group_dstrbs)

    def compute_ij(gi, gj, i, j):
        d = sliced_wasserstein_distance(
            gi, gj, ground_metric, p=p, n_projections=n_projections
        )
        return (i, j, d)

    tasks = [(i, j) for i in range(n) for j in range(i + 1, n)]

    D = np.zeros((n, n))
    for i in tqdm(range(0, len(tasks), batch_size), desc="Sliced Wasserstein Batches"):
        batch_tasks = tasks[i : i + batch_size]
        batch_results = Parallel(n_jobs=nCores)(
            delayed(compute_ij)(
                group_dstrbs[task[0]], group_dstrbs[task[1]], task[0], task[1]
            )
            for task in batch_tasks
        )

        for i, j, d in batch_results:
            D[i, j] = d
            D[j, i] = d

    return D


def wasserstein_distances_optimized(
    df_enc,
    method="sinkhorn",
    ground_metric="cosine",
    p=2,
    parallel=True,
    nCores=-1,
    batch_size=10,
    n_projections=50,
    reg=0.1,
    pca_dim=None,
    sparse_samples=None,
):
    """
    Optimized Wasserstein distance with multiple performance enhancements:
    - PCA dimension reduction before distance computation
    - Sparse approximation by subsampling points
    - Choice of algorithm: 'exact', 'sinkhorn', or 'sliced'
    - Efficient parallelization with batching
    """
    dim_cols = [c for c in df_enc.columns if c.startswith("dim_")]
    group_dstrbs = [group[dim_cols].values for _, group in df_enc.groupby("review_id")]
    n = len(group_dstrbs)

    if pca_dim is not None and pca_dim > 0:
        all_points = np.vstack(group_dstrbs)
        pca = PCA(n_components=min(pca_dim, all_points.shape[1]))
        all_points_reduced = pca.fit_transform(all_points)

        offset = 0
        group_dstrbs_reduced = []
        for g in group_dstrbs:
            next_offset = offset + g.shape[0]
            group_dstrbs_reduced.append(all_points_reduced[offset:next_offset])
            offset = next_offset

        group_dstrbs = group_dstrbs_reduced

    if sparse_samples is not None and sparse_samples > 0:
        group_dstrbs_sparse = []
        for g in group_dstrbs:
            if g.shape[0] > sparse_samples:
                idxs = np.random.choice(g.shape[0], sparse_samples, replace=False)
                group_dstrbs_sparse.append(g[idxs])
            else:
                group_dstrbs_sparse.append(g)

        group_dstrbs = group_dstrbs_sparse

    if method == "exact":
        distance_func = wasserstein_distance
    elif method == "sinkhorn":

        def distance_func(X, Y, ground_metric, p):
            return wasserstein_distance_sinkhorn(X, Y, ground_metric, p, reg=reg)

    elif method == "sliced":

        def distance_func(X, Y, ground_metric, p):
            return sliced_wasserstein_distance(
                X, Y, ground_metric, p, n_projections=n_projections
            )

    else:
        raise ValueError(f"Unknown method: {method}")

    if parallel:

        def compute_ij(gi, gj, i, j):
            d = distance_func(gi, gj, ground_metric, p)
            return (i, j, d)

        tasks = [(i, j) for i in range(n) for j in range(i + 1, n)]

        D = np.zeros((n, n))
        for i in tqdm(
            range(0, len(tasks), batch_size),
            desc=f"{method.capitalize()} Wasserstein Batches",
        ):
            batch_tasks = tasks[i : i + batch_size]
            batch_results = Parallel(n_jobs=nCores)(
                delayed(compute_ij)(
                    group_dstrbs[task[0]], group_dstrbs[task[1]], task[0], task[1]
                )
                for task in batch_tasks
            )

            for i, j, d in batch_results:
                D[i, j] = d
                D[j, i] = d
    else:
        D = np.zeros((n, n))
        for i in tqdm(range(n), desc=f"{method.capitalize()} Wasserstein"):
            for j in range(i + 1, n):
                dist_ij = distance_func(
                    group_dstrbs[i], group_dstrbs[j], ground_metric, p
                )
                D[i, j] = dist_ij
                D[j, i] = dist_ij

    return D


def geodesic_isomap(D: np.ndarray, k: int) -> np.ndarray:
    """
    Given full distance matrix D and neighborhood size k,
    builds the k‑NN graph (weighted by distances), symmetrizes it,
    and returns the all‑pairs geodesic distances.
    """
    G = kneighbors_graph(
        D, 
        n_neighbors=k, 
        mode='distance', 
        metric='precomputed', 
        include_self=False
    )
    G = G.minimum(G.T)
    return shortest_path(G, directed=False)