"""
BERT Representation Analysis using Topological Data Analysis

This script analyzes how persistence diagrams differ for various stratified groups
of movie reviews encoded with BERT. The analysis includes:
1. Loading stratified data and computing BERT encodings
2. Computing distance matrices using different metrics
3. Generating persistence diagrams for different groups
4. Comparing diagrams across groups using various metrics
5. Visualizing results with persistence diagrams, landscapes, and Betti curves
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from tomato.utils import tomato_data_path
from tomato.encoding import bert_encode_reviews
from tomato.metrics import wasserstein_distances_optimized, geodesic_isomap
from tomato.persistence_analysis import PersistenceAnalyzer
from tomato.tda import witness_dm

RESULTS_DIR = project_root / "results" / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ENCODINGS_FILE = RESULTS_DIR / "bert_encodings_strat800.parquet"
DISTANCE_MATRICES_FILE = RESULTS_DIR / "distance_matrices_strat800.npz"
PERSISTENCE_RESULTS_FILE = RESULTS_DIR / "persistence_results_strat800.json"
PERSISTENCE_DIAGRAMS_FILE = RESULTS_DIR / "persistence_diagrams_strat800.npz"


def load_stratified_data():
    print("Loading stratified data...")
    strat_file = tomato_data_path() / "df_strat800.csv"
    if not strat_file.exists():
        raise FileNotFoundError(f"Stratified data file not found: {strat_file}")
    df_strat = pd.read_csv(strat_file)
    print(f"Loaded {len(df_strat)} stratified samples")
    group_counts = (
        df_strat.groupby(
            ["critic_name", "content_rating", "drama_or_comedy", "review_type"]
        )
        .size()
        .reset_index(name="count")
    )
    print("Group distribution:")
    print(group_counts)
    return df_strat


def compute_bert_encodings(df_strat):
    print("Checking for cached BERT encodings...")
    if ENCODINGS_FILE.exists():
        print(f"Loading cached encodings from {ENCODINGS_FILE}")
        return pd.read_parquet(ENCODINGS_FILE)
    print("Computing BERT encodings for all samples...")
    texts = df_strat.review_content.astype(str).tolist()
    if "review_id" in df_strat.columns:
        ids = df_strat.review_id.tolist()
    else:
        ids = df_strat.index.tolist()
    df_enc = bert_encode_reviews(texts, ids, "bert-base-uncased")
    print(f"Saving encodings to {ENCODINGS_FILE}")
    df_enc.to_parquet(ENCODINGS_FILE, compression="snappy")
    return df_enc


def compute_distance_matrices(df_enc, df_strat):
    print("Checking for cached distance matrices...")
    if DISTANCE_MATRICES_FILE.exists():
        print(f"Loading cached distance matrices from {DISTANCE_MATRICES_FILE}")
        return np.load(DISTANCE_MATRICES_FILE, allow_pickle=True)
    print("Computing distance matrices...")
    metric_params = [
        # Algorithm, method name, ground metric, other params
        # ("exact", "wasserstein_exact", "cosine", {}),
        ("sinkhorn", "wasserstein_sinkhorn", "cosine", {"reg": 0.1}),
        # ("sliced", "wasserstein_sliced", "cosine", {"n_projections": 50}),
        # ("sinkhorn", "wasserstein_sinkhorn_pca", "cosine", {"reg": 0.1, "pca_dim": 50}),
    ]
    distance_matrices = {}
    for algorithm, method_name, ground_metric, params in tqdm(
        metric_params, desc="Computing metrics"
    ):
        print(f"Computing {method_name}...")
        distance_matrix = wasserstein_distances_optimized(
            df_enc,
            method=algorithm,
            ground_metric=ground_metric,
            parallel=True,
            nCores=-1,
            batch_size=10,
            **params,
        )
        distance_matrices[method_name] = distance_matrix
    print("Computing geodesic distances...")
    base_matrix = distance_matrices["wasserstein_sinkhorn"]
    distance_matrices["geodesic_k10"] = geodesic_isomap(base_matrix, k=10)
    distance_matrices["geodesic_k20"] = geodesic_isomap(base_matrix, k=20)
    print(f"Saving distance matrices to {DISTANCE_MATRICES_FILE}")
    np.savez_compressed(DISTANCE_MATRICES_FILE, **distance_matrices)
    return distance_matrices


def create_group_distance_matrices(distance_matrices, df_strat):
    print("Creating group-specific distance matrices...")
    group_criteria = [
        ("critic_name", ["Dennis Schwartz", "Roger Ebert"]),
        ("content_rating", ["PG", "R"]),
        ("drama_or_comedy", ["Comedy", "Drama"]),
        ("review_type", ["Fresh", "Rotten"]),
        (
            "critic_genre",
            [
                "Dennis Schwartz_Comedy",
                "Dennis Schwartz_Drama",
                "Roger Ebert_Comedy",
                "Roger Ebert_Drama",
            ],
        ),
        ("rating_genre", ["PG_Comedy", "PG_Drama", "R_Comedy", "R_Drama"]),
        (
            "critic_rating",
            [
                "Dennis Schwartz_PG",
                "Dennis Schwartz_R",
                "Roger Ebert_PG",
                "Roger Ebert_R",
            ],
        ),
        (
            "full_strat",
            [
                f"{c}_{r}_{g}_{s}"
                for c in ["Dennis Schwartz", "Roger Ebert"]
                for r in ["PG", "R"]
                for g in ["Comedy", "Drama"]
                for s in ["Fresh", "Rotten"]
            ],
        ),
    ]
    df_strat["critic_genre"] = (
        df_strat["critic_name"] + "_" + df_strat["drama_or_comedy"]
    )
    df_strat["rating_genre"] = (
        df_strat["content_rating"] + "_" + df_strat["drama_or_comedy"]
    )
    df_strat["critic_rating"] = (
        df_strat["critic_name"] + "_" + df_strat["content_rating"]
    )
    df_strat["full_strat"] = (
        df_strat["critic_name"]
        + "_"
        + df_strat["content_rating"]
        + "_"
        + df_strat["drama_or_comedy"]
        + "_"
        + df_strat["review_type"]
    )
    group_distance_matrices = {}
    main_metric = "wasserstein_sinkhorn"
    metric_matrix = distance_matrices[main_metric]
    for criterion, groups in group_criteria:
        print(f"Processing groups for {criterion}...")
        group_indices = {
            group: np.where(df_strat[criterion] == group)[0] for group in groups
        }
        group_matrices = {}
        for group, indices in group_indices.items():
            if len(indices) > 0:
                group_matrix = metric_matrix[np.ix_(indices, indices)]
                group_matrices[group] = group_matrix
        group_distance_matrices[criterion] = group_matrices
    return group_distance_matrices


def compute_persistence_diagrams(group_distance_matrices):
    print("Checking for cached persistence diagrams...")
    if PERSISTENCE_DIAGRAMS_FILE.exists() and PERSISTENCE_RESULTS_FILE.exists():
        print(f"Loading cached persistence diagrams from {PERSISTENCE_DIAGRAMS_FILE}")
        diagrams_data = np.load(PERSISTENCE_DIAGRAMS_FILE, allow_pickle=True)
        print(f"Loading cached persistence results from {PERSISTENCE_RESULTS_FILE}")
        with open(PERSISTENCE_RESULTS_FILE, "r") as f:
            persistence_results = json.load(f)
        return diagrams_data, persistence_results
    print("Computing persistence diagrams for all groups...")
    all_diagrams = {}
    persistence_results = {}
    for criterion, group_matrices in group_distance_matrices.items():
        print(f"Computing persistence diagrams for {criterion}...")
        analyzer = PersistenceAnalyzer(max_homology_dim=2)
        analyzer.compute_persistence_diagrams(group_matrices, method="ripser", maxdim=2)
        analyzer.compute_all_landscapes(num_landscapes=5, resolution=100)
        analyzer.compute_all_betti_curves(resolution=100)
        analyzer.compute_all_statistics()
        bottleneck_distances = analyzer.compare_groups(metric="bottleneck")
        wasserstein_distances = analyzer.compare_groups(metric="wasserstein", p=2)
        criterion_results = {
            "bottleneck_distances": bottleneck_distances.to_dict(),
            "wasserstein_distances": wasserstein_distances.to_dict(),
            "statistics": analyzer.statistics,
        }
        persistence_results[criterion] = criterion_results
        all_diagrams[criterion] = analyzer.diagrams
        save_persistence_plots(analyzer, criterion)
    print(f"Saving persistence results to {PERSISTENCE_RESULTS_FILE}")
    with open(PERSISTENCE_RESULTS_FILE, "w") as f:
        json.dump(persistence_results, f)
    print(f"Saving persistence diagrams to {PERSISTENCE_DIAGRAMS_FILE}")
    np.savez_compressed(PERSISTENCE_DIAGRAMS_FILE, **all_diagrams)
    return all_diagrams, persistence_results


def save_persistence_plots(analyzer, criterion):
    plot_dir = RESULTS_DIR / "plots" / criterion
    plot_dir.mkdir(parents=True, exist_ok=True)
    group_names = list(analyzer.diagrams.keys())
    for group in group_names:
        fig = analyzer.plot_diagrams(group)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{group}_diagram.png", dpi=300)
        plt.close(fig)
    for dim in [0, 1, 2]:
        try:
            fig = analyzer.plot_landscapes(group_names=group_names, homology_dim=dim)
            plt.tight_layout()
            plt.savefig(plot_dir / f"landscapes_H{dim}.png", dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"Error creating landscape plot for H{dim}: {e}")
    for dim in [0, 1, 2]:
        try:
            fig = analyzer.plot_betti_curves(group_names=group_names, homology_dim=dim)
            plt.tight_layout()
            plt.savefig(plot_dir / f"betti_curves_H{dim}.png", dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"Error creating Betti curve plot for H{dim}: {e}")


def analyze_persistence_results(persistence_results):
    print("Analyzing persistence results...")
    summary_file = RESULTS_DIR / "persistence_summary.txt"
    print("Generating comparative visualizations...")
    create_comparative_visualizations(persistence_results)
    with open(summary_file, "w") as f:
        f.write("# BERT Representation Analysis with TDA\n")
        f.write(f"# Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Visualizations\n")
        f.write(
            "The following visualizations have been generated in the results/data/plots directory:\n\n"
        )
        f.write("### Individual Group Visualizations\n")
        f.write(
            "- Persistence Diagrams: Shows birth-death pairs of topological features\n"
        )
        f.write(
            "- Persistence Landscapes: Alternative representation of persistence diagrams\n"
        )
        f.write(
            "- Betti Curves: Shows how Betti numbers change with filtration value\n\n"
        )
        f.write("### Comparative Visualizations\n")
        f.write("- Distance Heatmaps: Visualizing distances between different groups\n")
        f.write("- MDS Plots: 2D projection of groups based on topological distances\n")
        f.write(
            "- Statistical Comparisons: Boxplots of persistence statistics by group\n\n"
        )
        for criterion, results in persistence_results.items():
            f.write(f"\n## {criterion.upper()} ANALYSIS\n")
            f.write(
                f"\nVisualizations for {criterion} can be found in: results/data/plots/{criterion}/\n"
            )
            f.write(
                f"Comparative visualizations are in: results/data/plots/comparative/{criterion}/\n\n"
            )
            f.write("\n### Wasserstein Distances Between Groups\n")
            wasserstein_df = pd.DataFrame(results["wasserstein_distances"])
            f.write(wasserstein_df.to_string() + "\n")
            f.write("\n### Bottleneck Distances Between Groups\n")
            bottleneck_df = pd.DataFrame(results["bottleneck_distances"])
            f.write(bottleneck_df.to_string() + "\n")
            f.write("\n### Topological Statistics\n")
            for group, stats in results["statistics"].items():
                f.write(f"\n#### {group}\n")
                for stat_name, stat_value in stats.items():
                    f.write(f"{stat_name}: {stat_value}\n")
    print(f"Summary written to {summary_file}")


def create_comparative_visualizations(persistence_results):
    print("Creating comparative visualizations...")
    comp_dir = RESULTS_DIR / "plots" / "comparative"
    comp_dir.mkdir(parents=True, exist_ok=True)
    for criterion, results in persistence_results.items():
        print(f"Creating comparative visualizations for {criterion}...")
        criterion_dir = comp_dir / criterion
        criterion_dir.mkdir(parents=True, exist_ok=True)
        try:
            wasserstein_df = pd.DataFrame(results["wasserstein_distances"])
            bottleneck_df = pd.DataFrame(results["bottleneck_distances"])
            create_distance_heatmaps(
                wasserstein_df, bottleneck_df, criterion_dir, criterion
            )
            create_mds_plots(wasserstein_df, bottleneck_df, criterion_dir, criterion)
            create_statistical_plots(results["statistics"], criterion_dir, criterion)
        except Exception as e:
            print(f"Error creating visualizations for {criterion}: {e}")


def create_distance_heatmaps(wasserstein_df, bottleneck_df, output_dir, criterion):
    print("Creating distance heatmaps...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(wasserstein_df, annot=True, cmap="YlGnBu", square=True)
    plt.title(f"Wasserstein Distances Between Groups - {criterion}")
    plt.tight_layout()
    plt.savefig(output_dir / "wasserstein_heatmap.png", dpi=300)
    plt.close()
    plt.figure(figsize=(10, 8))
    sns.heatmap(bottleneck_df, annot=True, cmap="YlGnBu", square=True)
    plt.title(f"Bottleneck Distances Between Groups - {criterion}")
    plt.tight_layout()
    plt.savefig(output_dir / "bottleneck_heatmap.png", dpi=300)
    plt.close()


def create_mds_plots(wasserstein_df, bottleneck_df, output_dir, criterion):
    print("Creating MDS plots...")
    from sklearn.manifold import MDS

    try:
        wasserstein_array = wasserstein_df.values.copy()
        if np.all(np.isnan(wasserstein_array) | np.isinf(wasserstein_array)):
            print(
                f"Cannot create MDS plot for {criterion}: All distances are NaN or infinite"
            )
            return
        finite_values = wasserstein_array[
            ~np.isnan(wasserstein_array) & ~np.isinf(wasserstein_array)
        ]
        if len(finite_values) > 0:
            max_finite = np.max(finite_values)
            replacement_value = max_finite * 10 if max_finite > 0 else 1000
            wasserstein_array[
                np.isinf(wasserstein_array) | np.isnan(wasserstein_array)
            ] = replacement_value
            wasserstein_array = (wasserstein_array + wasserstein_array.T) / 2
            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=42,
                max_iter=300,
                eps=1e-6,
                n_init=4,
            )
            positions = mds.fit_transform(wasserstein_array)
            plt.figure(figsize=(10, 8))
            plt.scatter(positions[:, 0], positions[:, 1], s=100)
            for i, label in enumerate(wasserstein_df.index):
                plt.annotate(
                    label,
                    (positions[i, 0], positions[i, 1]),
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                )
            plt.title(f"MDS of Wasserstein Distances - {criterion}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "wasserstein_mds.png", dpi=300)
            plt.close()
        else:
            print(
                f"Cannot create MDS plot for {criterion}: No finite values in distance matrix"
            )
    except Exception as e:
        print(f"Error creating MDS plot for {criterion}: {e}")


def create_statistical_plots(statistics, output_dir, criterion):
    print("Creating statistical plots...")
    if not statistics:
        return
    stats_data = []
    for group, stats in statistics.items():
        group_stats = {"group": group}
        group_stats.update(stats)
        stats_data.append(group_stats)
    stats_df = pd.DataFrame(stats_data)
    for col in stats_df.columns:
        if col != "group":
            stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce")
    plot_stats = [
        "dim0_num_features",
        "dim0_mean_persistence",
        "dim1_num_features",
        "dim1_mean_persistence",
        "dim2_num_features",
        "dim2_mean_persistence",
    ]
    plot_stats = [s for s in plot_stats if s in stats_df.columns]
    if not plot_stats:
        return
    n_stats = len(plot_stats)
    n_rows = (n_stats + 1) // 2  # Calculate number of rows needed
    plt.figure(figsize=(14, 4 * n_rows))
    for i, stat in enumerate(plot_stats):
        plt.subplot(n_rows, 2, i + 1)
        try:
            if len(stats_df) <= 10:
                plot_data = stats_df[["group", stat]].dropna()
                if not plot_data.empty:
                    sns.barplot(x="group", y=stat, data=plot_data)
                    plt.xticks(rotation=45, ha="right")
                    plt.title(f"{stat}")
            else:
                plot_data = stats_df[stat].dropna()
                if not plot_data.empty:
                    plt.boxplot(plot_data)
                    plt.title(f"{stat} Distribution")
            plt.tight_layout()
        except Exception as e:
            print(f"Error plotting {stat}: {e}")
    plt.suptitle(f"Statistical Comparison - {criterion}", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "statistics_comparison.png", dpi=300)
    plt.close()


def main():
    print("Starting BERT representation analysis with TDA...")
    df_strat = load_stratified_data()
    df_enc = compute_bert_encodings(df_strat)
    distance_matrices = compute_distance_matrices(df_enc, df_strat)
    group_distance_matrices = create_group_distance_matrices(
        distance_matrices, df_strat
    )
    diagrams, persistence_results = compute_persistence_diagrams(
        group_distance_matrices
    )
    analyze_persistence_results(persistence_results)
    print("Analysis complete! Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
