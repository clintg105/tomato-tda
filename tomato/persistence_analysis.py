import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import pandas as pd
import persim
from ripser import ripser
from persim import plot_diagrams

from tomato.tda import witness_dm


class PersistenceLandscape:
    def __init__(
        self, num_landscapes: int = 5, resolution: int = 100, domain: tuple = (0, 1)
    ):
        self.num_landscapes = num_landscapes
        self.resolution = resolution
        self.domain = domain

    def fit_transform(self, diagrams: List[np.ndarray]) -> np.ndarray:
        landscapes = np.zeros((self.num_landscapes, self.resolution))
        grid = np.linspace(self.domain[0], self.domain[1], self.resolution)
        for diagram in diagrams:
            if diagram.shape[0] == 0:
                continue
            for idx, t in enumerate(grid):
                values = [max(0, min(t - birth, death - t)) for birth, death in diagram]
                values.sort(reverse=True)
                for k in range(min(self.num_landscapes, len(values))):
                    landscapes[k, idx] = max(landscapes[k, idx], values[k])
        return landscapes

    def plot(
        self,
        landscapes: np.ndarray,
        title: str = "Persistence Landscape",
        ax=None,
        color=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        grid = np.linspace(self.domain[0], self.domain[1], self.resolution)
        for i in range(self.num_landscapes):
            if np.any(landscapes[i] > 0):
                ax.plot(grid, landscapes[i], label=f"λ_{i+1}", alpha=0.8, color=color)
        ax.set_xlabel("t")
        ax.set_ylabel("λ(t)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return ax


class BettiCurve:
    def __init__(self, resolution: int = 100, domain: tuple = (0, 1)):
        self.resolution = resolution
        self.domain = domain

    def fit_transform(self, diagrams: List[np.ndarray]) -> np.ndarray:
        grid = np.linspace(self.domain[0], self.domain[1], self.resolution)
        betti_curve = np.zeros(self.resolution)
        for diagram in diagrams:
            if diagram.shape[0] == 0:
                continue
            for idx, t in enumerate(grid):
                betti_curve[idx] += np.sum((diagram[:, 0] <= t) & (t < diagram[:, 1]))
        return betti_curve

    def plot(
        self, betti_curve: np.ndarray, title: str = "Betti Curve", ax=None, color=None
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        grid = np.linspace(self.domain[0], self.domain[1], self.resolution)
        ax.plot(grid, betti_curve, color=color)
        ax.set_xlabel("Filtration value")
        ax.set_ylabel("Betti number")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return ax


def compute_persistence_statistics(diagrams: List[np.ndarray]) -> Dict:
    stats = {}
    all_persistence = []
    for dim, diagram in enumerate(diagrams):
        if diagram.shape[0] == 0:
            continue
        persistence_values = diagram[:, 1] - diagram[:, 0]
        key_prefix = f"dim{dim}_"
        stats[f"{key_prefix}num_features"] = len(persistence_values)
        stats[f"{key_prefix}total_persistence"] = np.sum(persistence_values)
        stats[f"{key_prefix}max_persistence"] = (
            np.max(persistence_values) if len(persistence_values) > 0 else 0
        )
        stats[f"{key_prefix}mean_persistence"] = (
            np.mean(persistence_values) if len(persistence_values) > 0 else 0
        )
        stats[f"{key_prefix}median_persistence"] = (
            np.median(persistence_values) if len(persistence_values) > 0 else 0
        )
        stats[f"{key_prefix}std_persistence"] = (
            np.std(persistence_values) if len(persistence_values) > 0 else 0
        )
        all_persistence.extend(persistence_values)
    if all_persistence:
        stats["total_persistence"] = np.sum(all_persistence)
        stats["max_persistence"] = np.max(all_persistence)
        stats["mean_persistence"] = np.mean(all_persistence)
        stats["median_persistence"] = np.median(all_persistence)
        stats["std_persistence"] = np.std(all_persistence)
    return stats


class DiagramDistance:
    @staticmethod
    def wasserstein(diagram1: np.ndarray, diagram2: np.ndarray, p: int = 2) -> float:
        return persim.wasserstein(diagram1, diagram2, p)

    @staticmethod
    def bottleneck(diagram1: np.ndarray, diagram2: np.ndarray) -> float:
        return persim.bottleneck(diagram1, diagram2)


class PersistenceAnalyzer:
    def __init__(self, max_homology_dim: int = 1):
        self.max_homology_dim = max_homology_dim
        self.diagrams = {}
        self.landscapes = {}
        self.betti_curves = {}
        self.statistics = {}

    def compute_persistence_diagrams(
        self, distance_matrices: Dict[str, np.ndarray], method: str = "ripser", **kwargs
    ) -> Dict[str, List[np.ndarray]]:
        diagrams = {}
        for group_name, distance_matrix in distance_matrices.items():
            if method.lower() == "ripser":
                thresh = kwargs.get("thresh", np.max(distance_matrix) * 0.5)
                result = ripser(
                    distance_matrix,
                    maxdim=self.max_homology_dim,
                    thresh=thresh,
                    distance_matrix=True,
                )
                diagrams[group_name] = result["dgms"]
            elif method.lower() == "witness":
                k = kwargs.get("k", 100)
                alpha2 = kwargs.get("alpha2", 1.0)
                seed = kwargs.get("seed", None)
                diagrams[group_name] = witness_dm(
                    distance_matrix,
                    k=k,
                    maxdim=self.max_homology_dim,
                    alpha2=alpha2,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        self.diagrams.update(diagrams)
        return diagrams

    def compute_all_landscapes(
        self, num_landscapes: int = 5, resolution: int = 100, domain: tuple = (0, 1)
    ) -> Dict:
        landscape_calculator = PersistenceLandscape(
            num_landscapes=num_landscapes, resolution=resolution, domain=domain
        )
        landscapes = {}
        for group_name, diagrams in self.diagrams.items():
            group_landscapes = {}
            for dim, diagram in enumerate(diagrams):
                if dim > self.max_homology_dim:
                    continue
                group_landscapes[f"H{dim}"] = landscape_calculator.fit_transform(
                    [diagram]
                )
            landscapes[group_name] = group_landscapes
        self.landscapes = landscapes
        return landscapes

    def compute_all_betti_curves(
        self, resolution: int = 100, domain: tuple = (0, 1)
    ) -> Dict:
        betti_calculator = BettiCurve(resolution=resolution, domain=domain)
        betti_curves = {}
        for group_name, diagrams in self.diagrams.items():
            group_curves = {}
            for dim, diagram in enumerate(diagrams):
                if dim > self.max_homology_dim:
                    continue
                group_curves[f"H{dim}"] = betti_calculator.fit_transform([diagram])
            betti_curves[group_name] = group_curves
        self.betti_curves = betti_curves
        return betti_curves

    def compute_all_statistics(self) -> Dict:
        statistics = {}
        for group_name, diagrams in self.diagrams.items():
            statistics[group_name] = compute_persistence_statistics(diagrams)
        self.statistics = statistics
        return statistics

    def compare_groups(self, metric: str = "wasserstein", p: int = 2) -> pd.DataFrame:
        group_names = list(self.diagrams.keys())
        n_groups = len(group_names)
        distances = np.zeros((n_groups, n_groups))
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                group1, group2 = group_names[i], group_names[j]
                dim_distances = []
                for dim in range(self.max_homology_dim + 1):
                    if dim < len(self.diagrams[group1]) and dim < len(
                        self.diagrams[group2]
                    ):
                        dgm1, dgm2 = (
                            self.diagrams[group1][dim],
                            self.diagrams[group2][dim],
                        )
                        if dgm1.size > 0 and dgm2.size > 0:
                            if metric == "wasserstein":
                                dist = DiagramDistance.wasserstein(dgm1, dgm2, p=p)
                            elif metric == "bottleneck":
                                dist = DiagramDistance.bottleneck(dgm1, dgm2)
                            else:
                                raise ValueError(f"Unknown metric: {metric}")
                            dim_distances.append(dist)
                if dim_distances:
                    numeric_distances = []
                    for d in dim_distances:
                        try:
                            if hasattr(d, "item") and callable(getattr(d, "item")):
                                numeric_distances.append(float(d.item()))
                            else:
                                numeric_distances.append(float(d))
                        except (ValueError, TypeError):
                            continue
                    if numeric_distances:
                        avg_dist = np.mean(numeric_distances)
                        distances[i, j] = avg_dist
                        distances[j, i] = avg_dist
        df_distances = pd.DataFrame(distances, index=group_names, columns=group_names)
        return df_distances

    def plot_diagrams(self, group_name: str, figsize: tuple = (10, 4)) -> plt.Figure:
        if group_name not in self.diagrams:
            raise ValueError(f"Group '{group_name}' not found")
        fig, ax = plt.subplots(figsize=figsize)
        plot_diagrams(self.diagrams[group_name], ax=ax)
        ax.set_title(f"Persistence Diagram: {group_name}")
        return fig

    def plot_landscapes(
        self,
        group_names: List[str] = None,
        homology_dim: int = 1,
        figsize: tuple = (10, 6),
    ) -> plt.Figure:
        if group_names is None:
            group_names = list(self.landscapes.keys())
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.tab10.colors
        for i, group_name in enumerate(group_names):
            if group_name in self.landscapes:
                key = f"H{homology_dim}"
                if key in self.landscapes[group_name]:
                    landscape = self.landscapes[group_name][key]
                    pl = PersistenceLandscape()
                    pl.resolution = landscape.shape[1]
                    pl.domain = (0, 1)
                    pl.plot(landscape, ax=ax, color=colors[i % len(colors)])
        ax.set_title(f"Persistence Landscapes (H{homology_dim})")
        ax.legend(group_names)
        return fig

    def plot_betti_curves(
        self,
        group_names: List[str] = None,
        homology_dim: int = 1,
        figsize: tuple = (10, 6),
    ) -> plt.Figure:
        if group_names is None:
            group_names = list(self.betti_curves.keys())
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.tab10.colors
        for i, group_name in enumerate(group_names):
            if group_name in self.betti_curves:
                key = f"H{homology_dim}"
                if key in self.betti_curves[group_name]:
                    betti_curve = self.betti_curves[group_name][key]
                    bc = BettiCurve()
                    bc.resolution = len(betti_curve)
                    bc.domain = (0, 1)
                    bc.plot(betti_curve, ax=ax, color=colors[i % len(colors)])
        ax.set_title(f"Betti Curves (H{homology_dim})")
        ax.legend(group_names)
        return fig
