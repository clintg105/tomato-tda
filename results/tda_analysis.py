#!/usr/bin/env python
"""
TDA Analysis script for movie review data
Analyzes topological structures in precomputed persistence diagrams
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from scipy.stats import wasserstein_distance
import json
from collections import defaultdict
import warnings

# Project imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tomato.database import TDAManager

# Set up paths
DATA_DIR = Path("/Users/lianghaochen/tomato-tda/data")
RESULTS_DIR = Path("/Users/lianghaochen/tomato-tda/results/data")
PLOTS_DIR = Path("/Users/lianghaochen/tomato-tda/results/data/plots")

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Datasets and metrics to analyze
DATASETS = ["bert_strat800", "tfidf_strat800", "bow_strat800"]
REDUCTIONS = ["pooled"]
BASE_METRICS = ["cos", "mp1"] 
SPLITS = ["Dennis Schwartz", "Roger Ebert", "Fresh", "Rotten", "PG", "R", "Comedy", "Drama"]

# Initialize TDA Manager
tm = TDAManager()

def load_persistence_diagram(dataset, reduction, metric, split):
    """Load a persistence diagram from the precomputed NPZ files"""
    try:
        npz_file = DATA_DIR / dataset / reduction / "tda" / "split" / split / f"{metric}.npz"
        if not npz_file.exists():
            return None
        data = np.load(npz_file, allow_pickle=True)
        return data['out'].item()
    except Exception as e:
        warnings.warn(f"Error loading {npz_file}: {e}")
        return None

def persistence_statistics(dgms):
    """Calculate statistics from persistence diagrams"""
    stats = {}
    
    # For each homology dimension (H0, H1, H2)
    for dim, dgm in enumerate(dgms):
        if dgm.size == 0:
            stats[f"H{dim}_count"] = 0
            stats[f"H{dim}_max_persistence"] = 0
            stats[f"H{dim}_total_persistence"] = 0
            stats[f"H{dim}_persistence_entropy"] = 0
            stats[f"H{dim}_avg_persistence"] = 0
            continue
            
        # Number of features
        stats[f"H{dim}_count"] = len(dgm)
        
        # Persistence values (birth-death pairs)
        persistence = dgm[:, 1] - dgm[:, 0]
        
        # Filter out infinite values
        finite_persistence = persistence[np.isfinite(persistence)]
        if len(finite_persistence) == 0:
            stats[f"H{dim}_max_persistence"] = 0
            stats[f"H{dim}_total_persistence"] = 0
            stats[f"H{dim}_persistence_entropy"] = 0
            stats[f"H{dim}_avg_persistence"] = 0
            continue
            
        # Max persistence
        stats[f"H{dim}_max_persistence"] = np.max(finite_persistence)
        
        # Total persistence
        stats[f"H{dim}_total_persistence"] = np.sum(finite_persistence)
        
        # Average persistence
        stats[f"H{dim}_avg_persistence"] = np.mean(finite_persistence)
        
        # Persistence entropy (normalized)
        if np.sum(finite_persistence) > 0:
            p = finite_persistence / np.sum(finite_persistence)
            entropy = -np.sum(p * np.log(p + 1e-10))
            stats[f"H{dim}_persistence_entropy"] = entropy
        else:
            stats[f"H{dim}_persistence_entropy"] = 0
            
    return stats

def bottleneck_distance(dgm1, dgm2, dim=1):
    """Compute bottleneck distance between persistence diagrams"""
    from scipy.spatial.distance import directed_hausdorff
    
    # Extract the diagrams for the requested dimension
    if dim >= len(dgm1) or dim >= len(dgm2) or dgm1[dim].size == 0 or dgm2[dim].size == 0:
        return np.nan
        
    diag1 = dgm1[dim]
    diag2 = dgm2[dim]
    
    # Use directed Hausdorff distance as approximation
    dist1, _, _ = directed_hausdorff(diag1, diag2)
    dist2, _, _ = directed_hausdorff(diag2, diag1)
    
    return max(dist1, dist2)

def analyze_encoding_topological_differences():
    """Compare topological structures across different encoding methods"""
    results = {}
    
    for split in SPLITS:
        encoding_stats = {}
        
        for dataset in DATASETS:
            for metric in BASE_METRICS:
                pd_data = load_persistence_diagram(dataset, "pooled", metric, split)
                if pd_data is None:
                    continue
                
                # Calculate statistics from diagrams
                stats = persistence_statistics(pd_data['dgms'])
                encoding_stats[f"{dataset}_{metric}"] = stats
                
        results[split] = encoding_stats
    
    # Save results
    with open(RESULTS_DIR / "encoding_topological_differences.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def analyze_critic_style_influence():
    """Analyze how critic writing style influences topological features"""
    critic_results = {}
    
    for dataset in DATASETS:
        for metric in BASE_METRICS:
            # Load data for both critics
            dennis_data = load_persistence_diagram(dataset, "pooled", metric, "Dennis Schwartz")
            roger_data = load_persistence_diagram(dataset, "pooled", metric, "Roger Ebert")
            
            if dennis_data is None or roger_data is None:
                continue
                
            # Calculate bottleneck distances between critics for each homology dimension
            bottleneck_dists = {}
            for dim in range(len(dennis_data['dgms'])):
                dist = bottleneck_distance(dennis_data['dgms'], roger_data['dgms'], dim)
                bottleneck_dists[f"H{dim}"] = dist
            
            critic_results[f"{dataset}_{metric}"] = bottleneck_dists
    
    # Save results
    with open(RESULTS_DIR / "critic_topological_comparison.json", "w") as f:
        json.dump(critic_results, f, indent=2)
    
    return critic_results

def analyze_sentiment_topology():
    """Analyze topological differences between Fresh and Rotten reviews"""
    sentiment_results = {}
    
    for dataset in DATASETS:
        for metric in BASE_METRICS:
            # Load data for Fresh and Rotten reviews
            fresh_data = load_persistence_diagram(dataset, "pooled", metric, "Fresh")
            rotten_data = load_persistence_diagram(dataset, "pooled", metric, "Rotten")
            
            if fresh_data is None or rotten_data is None:
                continue
                
            # Calculate statistics
            fresh_stats = persistence_statistics(fresh_data['dgms'])
            rotten_stats = persistence_statistics(rotten_data['dgms'])
            
            # Compare relative feature density between Fresh/Rotten
            feature_density = {}
            for dim in range(len(fresh_data['dgms'])):
                fresh_count = fresh_stats[f"H{dim}_count"]
                rotten_count = rotten_stats[f"H{dim}_count"]
                if fresh_count > 0 and rotten_count > 0:
                    ratio = fresh_count / rotten_count
                    feature_density[f"H{dim}_fresh_rotten_ratio"] = ratio
                    
                # Compare entropy differences
                entropy_diff = fresh_stats[f"H{dim}_persistence_entropy"] - rotten_stats[f"H{dim}_persistence_entropy"]
                feature_density[f"H{dim}_entropy_diff"] = entropy_diff
            
            sentiment_results[f"{dataset}_{metric}"] = feature_density
    
    # Save results
    with open(RESULTS_DIR / "sentiment_topology.json", "w") as f:
        json.dump(sentiment_results, f, indent=2)
    
    return sentiment_results

def analyze_content_rating_topology():
    """Analyze topological differences between PG and R rated movies"""
    rating_results = {}
    
    for dataset in DATASETS:
        for metric in BASE_METRICS:
            # Load data for PG and R ratings
            pg_data = load_persistence_diagram(dataset, "pooled", metric, "PG")
            r_data = load_persistence_diagram(dataset, "pooled", metric, "R")
            
            if pg_data is None or r_data is None:
                continue
            
            # Calculate statistics
            pg_stats = persistence_statistics(pg_data['dgms'])
            r_stats = persistence_statistics(r_data['dgms'])
            
            # Compare persistence lifetimes
            lifetime_comparison = {}
            for dim in range(min(len(pg_data['dgms']), len(r_data['dgms']))):
                pg_lifetime = pg_stats[f"H{dim}_avg_persistence"]
                r_lifetime = r_stats[f"H{dim}_avg_persistence"]
                
                if pg_lifetime > 0 and r_lifetime > 0:
                    lifetime_ratio = pg_lifetime / r_lifetime
                    lifetime_comparison[f"H{dim}_lifetime_ratio"] = lifetime_ratio
                
                # Compare maximum persistence
                pg_max = pg_stats[f"H{dim}_max_persistence"]
                r_max = r_stats[f"H{dim}_max_persistence"]
                lifetime_comparison[f"H{dim}_max_persistence_diff"] = pg_max - r_max
            
            rating_results[f"{dataset}_{metric}"] = lifetime_comparison
    
    # Save results
    with open(RESULTS_DIR / "rating_topology.json", "w") as f:
        json.dump(rating_results, f, indent=2)
    
    return rating_results

def calculate_topological_complexity():
    """Calculate topological complexity scores for each split and encoding"""
    complexity_scores = {}
    
    for dataset in DATASETS:
        complexity_scores[dataset] = {}
        
        for split in SPLITS:
            split_scores = {}
            
            for metric in BASE_METRICS:
                pd_data = load_persistence_diagram(dataset, "pooled", metric, split)
                if pd_data is None:
                    continue
                
                # Calculate a complexity score based on persistent homology
                # Higher score = more complex topology
                complexity = 0
                for dim, dgm in enumerate(pd_data['dgms']):
                    if dgm.size == 0:
                        continue
                        
                    # Calculate persistence
                    persistence = dgm[:, 1] - dgm[:, 0]
                    
                    # Filter out infinite values
                    finite_persistence = persistence[np.isfinite(persistence)]
                    if len(finite_persistence) == 0:
                        continue
                    
                    # Higher dimensions contribute more to complexity
                    dim_weight = dim + 1
                    
                    # Complexity combines feature count, persistence and dimension
                    dimension_complexity = (
                        len(finite_persistence) * 
                        np.sum(finite_persistence) * 
                        dim_weight
                    )
                    
                    complexity += dimension_complexity
                
                split_scores[metric] = complexity
            
            complexity_scores[dataset][split] = split_scores
    
    # Save results
    with open(RESULTS_DIR / "topological_complexity.json", "w") as f:
        json.dump(complexity_scores, f, indent=2)
    
    return complexity_scores

def visualize_persistence_diagrams():
    """Create visualizations for selected persistence diagrams"""
    # Choose representative examples
    examples = [
        ("bert_strat800", "pooled", "cos", "Drama"),
        ("bert_strat800", "pooled", "cos", "Comedy"),
        ("bert_strat800", "pooled", "cos", "Fresh"),
        ("bert_strat800", "pooled", "cos", "Rotten"),
        ("tfidf_strat800", "pooled", "cos", "Fresh"),
        ("bow_strat800", "pooled", "cos", "Fresh"),
    ]
    
    # Create comparison plots for each example
    for dataset, reduction, metric, split in examples:
        pd_data = load_persistence_diagram(dataset, reduction, metric, split)
        if pd_data is None:
            continue
        
        dgms = pd_data['dgms']
        
        # Create a persistence diagram plot
        fig, axes = plt.subplots(1, min(3, len(dgms)), figsize=(15, 5))
        if len(dgms) == 1:
            axes = [axes]
        
        for dim, (ax, dgm) in enumerate(zip(axes, dgms)):
            if dgm.size == 0:
                ax.set_title(f"H{dim} (empty)")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue
                
            # Plot points - filter out infinities and NaNs
            finite_mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
            if not np.any(finite_mask):
                ax.set_title(f"H{dim} (no finite points)")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue
                
            finite_dgm = dgm[finite_mask]
            ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], s=30, alpha=0.6, label=f"H{dim}")
            
            # Plot diagonal with safe limits
            min_val = finite_dgm[:, 0].min() if finite_dgm.size > 0 else 0
            max_val = finite_dgm[:, 1].max() if finite_dgm.size > 0 else 1
            margin = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
            lim_min = max(0, min_val - margin)
            lim_max = max_val + margin
            
            # Ensure limits are valid numbers
            if not np.isfinite(lim_min) or not np.isfinite(lim_max):
                lim_min, lim_max = 0, 1
            
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3)
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
            ax.set_title(f"H{dim} Persistence Diagram")
            ax.set_xlabel("Birth")
            ax.set_ylabel("Death")
            ax.grid(alpha=0.2)
        
        plt.suptitle(f"Persistence Diagrams: {dataset}, {metric}, {split}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"persistence_diagram_{dataset}_{metric}_{split.replace(' ', '_')}.png", dpi=300)
        plt.close()

def visualize_topological_complexity():
    """Visualize topological complexity scores"""
    # Load complexity scores
    with open(RESULTS_DIR / "topological_complexity.json", "r") as f:
        complexity_scores = json.load(f)
    
    # Prepare data for visualization
    plot_data = []
    for dataset in complexity_scores:
        for split in complexity_scores[dataset]:
            for metric in complexity_scores[dataset][split]:
                score = complexity_scores[dataset][split][metric]
                plot_data.append({
                    "Dataset": dataset.split("_")[0],  # Just the encoder name
                    "Split": split,
                    "Metric": metric,
                    "Complexity": score
                })
    
    df = pd.DataFrame(plot_data)
    
    # Create heatmap visualization
    pivot_data = df.pivot_table(
        values="Complexity", 
        index=["Dataset", "Metric"], 
        columns="Split",
        aggfunc="mean"
    )
    
    # Normalize within each row for better visualization
    normalized_data = pivot_data.div(pivot_data.max(axis=1), axis=0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(normalized_data, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
    plt.title("Normalized Topological Complexity Across Splits and Encodings")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "topological_complexity_heatmap.png", dpi=300)
    plt.close()
    
    # Also create a bar chart comparison
    plt.figure(figsize=(14, 10))
    
    # Average across metrics for simplicity
    avg_by_encoder_split = df.groupby(["Dataset", "Split"])["Complexity"].mean().reset_index()
    
    sns.barplot(x="Split", y="Complexity", hue="Dataset", data=avg_by_encoder_split)
    plt.yscale('log')  # Log scale for better visualization
    plt.title("Average Topological Complexity by Encoding Method and Split")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "topological_complexity_barplot.png", dpi=300)
    plt.close()

def analyze_language_complexity():
    """
    Analyze the relationship between language complexity and topological structure
    focusing on H1 persistence features, which often capture cyclic relationships
    """
    h1_features = {}
    
    for dataset in DATASETS:
        h1_features[dataset] = {}
        
        for split in SPLITS:
            for metric in BASE_METRICS:
                pd_data = load_persistence_diagram(dataset, "pooled", metric, split)
                if pd_data is None or len(pd_data['dgms']) < 2:
                    continue
                
                h1_dgm = pd_data['dgms'][1]
                if h1_dgm.size == 0:
                    continue
                    
                # Calculate persistence
                persistence = h1_dgm[:, 1] - h1_dgm[:, 0]
                
                # Filter out infinite values
                finite_persistence = persistence[np.isfinite(persistence)]
                if len(finite_persistence) == 0:
                    continue
                
                # Store H1 feature data
                h1_features[dataset][f"{split}_{metric}"] = {
                    "count": len(finite_persistence),
                    "avg_persistence": float(np.mean(finite_persistence)),
                    "max_persistence": float(np.max(finite_persistence)),
                    "persistence_distribution": finite_persistence.tolist()
                }
    
    # Save results
    with open(RESULTS_DIR / "h1_features.json", "w") as f:
        json.dump(h1_features, f, indent=2)
    
    # Create visualization comparing H1 features
    plot_data = []
    for dataset in h1_features:
        for key, data in h1_features[dataset].items():
            split = key.split('_')[0]
            plot_data.append({
                "Dataset": dataset.split("_")[0],
                "Split": split,
                "H1 Count": data["count"],
                "Avg Persistence": data["avg_persistence"]
            })
    
    df = pd.DataFrame(plot_data)
    
    # Scatterplot of H1 features
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x="H1 Count",
        y="Avg Persistence",
        hue="Dataset",
        style="Split",
        s=100,
        alpha=0.7
    )
    plt.title("H1 Homology Features: Count vs. Average Persistence")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h1_feature_comparison.png", dpi=300)
    plt.close()

def main():
    """Main analysis function"""
    print("Analyzing encoding topological differences...")
    analyze_encoding_topological_differences()
    
    print("Analyzing critic style influence...")
    analyze_critic_style_influence()
    
    print("Analyzing sentiment topology...")
    analyze_sentiment_topology()
    
    print("Analyzing content rating topology...")
    analyze_content_rating_topology()
    
    print("Calculating topological complexity...")
    calculate_topological_complexity()
    
    print("Analyzing language complexity...")
    analyze_language_complexity()
    
    print("Creating visualizations...")
    visualize_persistence_diagrams()
    visualize_topological_complexity()
    
    print("Analysis complete! Results saved to:", RESULTS_DIR)
    print("Plots saved to:", PLOTS_DIR)

if __name__ == "__main__":
    main()
