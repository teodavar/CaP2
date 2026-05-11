"""
comm_cost_variance_sweep.py

For each experiment folder, loads the fine-tuned model and partition info,
then estimates expected communication cost as a function of noise variance
added to the communication cost map (partition['maps']).

comm_cost = sum over cross-partition edges of: cost_map[i][j] * outsize
            (only for edges where at least one weight is nonzero)

Usage:
    python comm_cost_variance_sweep.py --logs_dir experiment_logs_test --n_samples 20

Dependencies (same as the original codebase):
    torch, numpy, matplotlib, yaml, re, os
    + the original source package (models, etc.)


To reload and replot in a future script — no model loading required:

    from comm_cost_variance_sweep import load_sweep_csv, plot_results

    results, variances, n_samples, seed = load_sweep_csv("experiment_logs_test/comm_cost_variance_sweep.csv")
    plot_results(results, variances, output_dir="./experiment_logs_test", n_samples=n_samples)


"""

import os
import re
import copy
import argparse
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── imports from the original codebase ─────────────────────────────────────
from source import models
from pareto import (          # <-- your existing file with all the helpers
    parse_experiment_folder,
    load_partition_data,
    compute_communication_cost_and_kbits,
)


# ═══════════════════════════════════════════════════════════════════════════
# Core sweep logic
# ═══════════════════════════════════════════════════════════════════════════

def sample_noisy_maps(original_maps, variance, n_samples, rng, clip_negative=True):
    """
    Draw `n_samples` cost-map realisations by adding zero-mean Gaussian noise
    with the given `variance` to every entry of `original_maps`.

    original_maps : list[list[float]]  — the partition['maps'] matrix
    Returns       : list of perturbed maps (same nested-list structure)
    """
    std = variance ** 0.5
    num_machines = len(original_maps)
    noisy_maps = []
    for _ in range(n_samples):
        new_map = []
        for i in range(num_machines):
            row = []
            for j in range(num_machines):
                noise = rng.normal(0.0, std)
                val = original_maps[i][j] + noise
                if clip_negative:
                    val = max(0.0, val)   # communication cost can't be negative
                row.append(val)
            new_map.append(row)
        noisy_maps.append(new_map)
    return noisy_maps


def expected_comm_cost_over_variances(model, partition, variances, n_samples=50, seed=42):
    """
    For each variance level, draw `n_samples` noisy cost maps, compute
    comm_cost for each, and return the mean and std across samples.

    Returns
    -------
    means : np.ndarray  shape (len(variances),)
    stds  : np.ndarray  shape (len(variances),)
    """
    rng = np.random.default_rng(seed)
    original_maps = partition["maps"]

    means, stds = [], []

    for var in variances:
        cost_samples = []

        if var == 0.0:
            # Exact value — no sampling needed
            comm_cost, _ = compute_communication_cost_and_kbits(model, partition)
            means.append(float(comm_cost))
            stds.append(0.0)
            continue

        noisy_maps_list = sample_noisy_maps(original_maps, var, n_samples, rng)

        for noisy_maps in noisy_maps_list:
            # Deep-copy partition so no nested object is shared, then swap maps
            part_copy = copy.deepcopy(partition)
            part_copy["maps"] = noisy_maps
            comm_cost, _ = compute_communication_cost_and_kbits(model, part_copy)
            cost_samples.append(float(comm_cost))

        means.append(float(np.mean(cost_samples)))
        stds.append(float(np.std(cost_samples)))

    return np.array(means), np.array(stds)


# ═══════════════════════════════════════════════════════════════════════════
# Experiment loader  (mirrors generate_visualizations, stripped down)
# ═══════════════════════════════════════════════════════════════════════════

def load_experiment(folder_path, experiment_details, device="cpu"):
    """
    Load model + partition for a single experiment folder.
    Returns (model, partition) or None if anything is missing.
    """
    model_state_path = os.path.join(folder_path, "fine_tuned.pt")
    if not os.path.exists(model_state_path):
        return None

    partition = load_partition_data(folder_path)
    if not partition:
        return None

    # ── build model ────────────────────────────────────────────────────────
    model_name = experiment_details["model"]
    try:
        if model_name == "resnet18":
            model = models.__dict__[model_name](
                nn.Conv2d, nn.BatchNorm2d, num_classes=10
            )
        elif model_name == "resnet101":
            from source.utils.misc import get_layers, get_bn_layers
            model = models.__dict__[model_name](
                get_layers("regular"), get_bn_layers("regular"), num_classes=100
            )
        else:
            print(f"  Unknown model '{model_name}', skipping.")
            return None
    except KeyError:
        print(f"  Model '{model_name}' not found in models registry, skipping.")
        return None

    model.load_state_dict(
        torch.load(model_state_path, map_location=torch.device(device))
    )
    model.eval()
    return model, partition


# ═══════════════════════════════════════════════════════════════════════════
# CSV persistence
# ═══════════════════════════════════════════════════════════════════════════

def save_results_csv(results, variances, n_samples, seed, output_dir):
    """
    Saves all sweep results to a tidy CSV with one row per (experiment, variance).

    Columns
    -------
    experiment      : human-readable label built from folder metadata
    data_code       : dataset identifier
    model           : model architecture
    num_partition   : number of machines
    pr_ratio        : pruning ratio
    sparsity_type   : sparsity strategy
    reassign_flag   : reassignment strategy
    n_samples       : Monte-Carlo samples used at each variance level
    seed            : RNG seed
    variance        : noise variance (σ²) applied to the cost map
    comm_cost_mean  : expected comm_cost at this variance
    comm_cost_std   : standard deviation of comm_cost across samples
    comm_cost_mean_minus_std : mean - std  (convenience for plotting bands)
    comm_cost_mean_plus_std  : mean + std  (convenience for plotting bands)

    The CSV can be reloaded later to reproduce any plot or table without
    re-running the expensive model loading + sampling sweep.
    """
    rows = []
    for label, (means, stds) in results.items():
        # Parse metadata back out of the label so each column is queryable
        parts = dict(p.split("=") if "=" in p else (p, p)
                     for p in label.replace(" ", "").split("|"))
        # label format: "data_code | model | np=X | pr=Y | sparsity_type-reassign_flag"
        # Easier to just store the raw label + split fields stored at run time
        for var, mean, std in zip(variances, means, stds):
            rows.append({
                "experiment":               label,
                "n_samples":                n_samples,
                "seed":                     seed,
                "variance":                 float(var),
                "comm_cost_mean":           float(mean),
                "comm_cost_std":            float(std),
                "comm_cost_mean_minus_std": float(mean - std),
                "comm_cost_mean_plus_std":  float(mean + std),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "comm_cost_variance_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅  Results saved → {csv_path}")
    return csv_path




def plot_results(results, variances, output_dir, n_samples):
    """
    results : dict  label -> (means, stds)
    """
    # ── aesthetics ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    n = len(results)
    cmap = cm.get_cmap("tab10", max(n, 1))
    colors = [cmap(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 6))

    for (label, (means, stds)), color in zip(results.items(), colors):
        ax.plot(variances, means, marker="o", linewidth=2,
                markersize=6, label=label, color=color)
        ax.fill_between(
            variances,
            means - stds,
            means + stds,
            alpha=0.15,
            color=color,
            label="_nolegend_",
        )

    ax.set_xlabel("Cost-map noise variance  (σ²)", fontsize=13)
    ax.set_ylabel("Expected comm_cost (weighted transfer count)", fontsize=13)
    ax.set_title(
        f"Expected comm_cost vs. cost-map noise variance\n"
        f"(shaded band = ±1 std,  {n_samples} samples per variance level)",
        fontsize=13,
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # ── legend: outside the axes, below the plot ───────────────────────────
    # Shorten labels: drop the fixed prefix fields shared by all experiments
    # and keep only the differentiating parts (pr, sparsity_type, reassign_flag)
    handles, raw_labels = ax.get_legend_handles_labels()
    short_labels = []
    for lbl in raw_labels:
        # label format: "data | model | np=X | pr=Y | sparsity-reassign"
        parts = [p.strip() for p in lbl.split("|")]
        # Keep pr=Y and the last field (sparsity-reassign); drop data/model/np
        # which are the same for all lines in a typical single-dataset run
        kept = [p for p in parts if p.startswith("pr=") or "|" not in p and parts.index(p) >= 3]
        # Fallback: just take last two pipe-separated fields
        short = " | ".join(parts[-2:]) if len(parts) >= 2 else lbl
        short_labels.append(short)

    legend = fig.legend(
        handles,
        short_labels,
        loc="lower center",
        fontsize=8,
        ncol=3,                     # spread across 3 columns
        bbox_to_anchor=(0.5, -0.02),
        framealpha=0.9,
        borderaxespad=0.5,
        handlelength=2,
    )

    plt.tight_layout(rect=[0, 0.22, 1, 1])   # reserve bottom 22% for legend
    save_path = os.path.join(output_dir, "comm_cost_vs_variance.png")
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n✅  Plot saved → {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_sweep(
    logs_dir,
    variances=None,
    n_samples=50,
    device="cpu",
    seed=42,
):
    """
    Batch over all experiment folders in `logs_dir`, compute the expected
    comm_cost vs. variance curve for each, save results to CSV, and plot.

    Parameters
    ----------
    logs_dir   : str   — path to the experiment logs directory
    variances  : list  — noise variance values to sweep (default: 0 → 10, 15 steps)
    n_samples  : int   — Monte-Carlo samples per variance level
    device     : str   — 'cpu' or 'cuda'
    seed       : int   — global RNG seed
    """
    if variances is None:
        variances = [0.0] + list(np.linspace(0.5, 10.0, 15))

    variances = np.array(variances)
    results = {}  # label -> (means, stds)

    for folder in sorted(os.listdir(logs_dir)):
        folder_path = os.path.join(logs_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        details = parse_experiment_folder(folder)
        if not details:
            print(f"[SKIP] Cannot parse folder name: {folder}")
            continue

        print(f"\n── Loading: {folder}")
        loaded = load_experiment(folder_path, details, device=device)
        if loaded is None:
            print(f"[SKIP] Missing model/partition in: {folder}")
            continue

        model, partition = loaded

        if "maps" not in partition:
            print(f"[SKIP] No 'maps' key in partition for: {folder}")
            continue

        label = (
            f"{details['data_code']} | {details['model']} | "
            f"np={details['num_partition']} | pr={details['pr_ratio']} | "
            f"{details['sparsity_type']}-{details['reassign_flag']}"
        )

        print(f"  → Sweeping {len(variances)} variance levels × {n_samples} samples …")
        means, stds = expected_comm_cost_over_variances(
            model, partition, variances, n_samples=n_samples, seed=seed
        )

        results[label] = (means, stds)
        print(f"  → comm_cost at σ²=0: {means[0]:.2f}  |  at σ²={variances[-1]:.1f}: {means[-1]:.2f}")

    if not results:
        print("\nNo valid experiments found — nothing to plot.")
        return

    save_results_csv(results, variances, n_samples, seed, logs_dir)
    plot_results(results, variances, logs_dir, n_samples)


# ═══════════════════════════════════════════════════════════════════════════
# CSV reload helper  (use this in future plotting/analysis scripts)
# ═══════════════════════════════════════════════════════════════════════════

def load_sweep_csv(csv_path):
    """
    Reload a previously saved sweep CSV and reconstruct the `results` dict
    and `variances` array needed by `plot_results`.

    Returns
    -------
    results   : dict  label -> (means np.ndarray, stds np.ndarray)
    variances : np.ndarray
    n_samples : int   (as recorded in the CSV)
    seed      : int   (as recorded in the CSV)

    Example
    -------
    results, variances, n_samples, seed = load_sweep_csv("comm_cost_variance_sweep.csv")
    plot_results(results, variances, output_dir=".", n_samples=n_samples)
    """
    df = pd.read_csv(csv_path)
    variances = df["variance"].unique()
    variances.sort()

    n_samples = int(df["n_samples"].iloc[0])
    seed      = int(df["seed"].iloc[0])

    results = {}
    for label, group in df.groupby("experiment"):
        group_sorted = group.sort_values("variance")
        means = group_sorted["comm_cost_mean"].to_numpy()
        stds  = group_sorted["comm_cost_std"].to_numpy()
        results[label] = (means, stds)

    return results, variances, n_samples, seed


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot expected MB transmitted vs. cost-map noise variance."
    )
    parser.add_argument(
        "--logs_dir",
        default="experiment_logs_uniform_10",
        help="Path to the experiment logs directory.",
    )
    parser.add_argument(
        "--var_min", type=float, default=0.5,
        help="Minimum non-zero variance value.",
    )
    parser.add_argument(
        "--var_max", type=float, default=10.0,
        help="Maximum variance value.",
    )
    parser.add_argument(
        "--var_steps", type=int, default=15,
        help="Number of variance levels between var_min and var_max.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=50,
        help="Monte-Carlo samples per variance level.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    variances = [0.0] + list(np.linspace(args.var_min, args.var_max, args.var_steps))

    run_sweep(
        logs_dir=args.logs_dir,
        variances=variances,
        n_samples=args.n_samples,
        device=args.device,
        seed=args.seed,
    )

#### To reload and replot in a future script — no model loading required:

'''
from comm_cost_variance_sweep import load_sweep_csv, plot_results

results, variances, n_samples, seed = load_sweep_csv("experiment_logs_test/comm_cost_variance_sweep.csv")
plot_results(results, variances, output_dir="./experiment_logs_test", n_samples=n_samples)
'''