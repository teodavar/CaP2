r"""
sensitivity_analysis.py
=======================

Fills in the sensitivity table from `dynamic-1-sensitivity.tex`
(Sec.: "Robustness to Communication-Cost Variation").

Given trained models (assumed pre-trained), evaluates the *realized*
communication cost when the deployment-time cost matrix C is replaced by a
perturbed matrix C'. Three perturbation families from the paper:

    1. Multiplicative log-normal noise   (σ ∈ {0.1, 0.25, 0.5, 1.0})
    2. Speed-class reshuffle             (p ∈ {0.1, 0.25, 0.5})
    3. Single-link spike                 (k ∈ {2, 5, 10})

Two metrics, both evaluated on C':
    - pencr : exact binary cost  (compute_communication_cost_and_kbits)
    - penbr : magnitude-weighted (compute_communication_loss)

Methods compared (each is a trained-model folder containing fine_tuned.pt
and partition_final):
    - CaMP-frozen  (paper macro: \Problemname)        --camp_dir
    - CaP-frozen   (paper macro: \Baseline)           --cap_dir
    - Dense-frozen (paper macro: \method{Dense})      --dense_dir (optional;
                    falls back to an analytic "every cross-machine edge
                    counts" cost when no dense checkpoint is provided)
    - CaMP-C'  (oracle, retrained from scratch under C') --oracle_root
                    (optional; oracle models are not yet trained -- the column
                    will be filled with "--" if no path is given)

Usage
-----
    python sensitivity_analysis.py \
        --camp_dir   <path>/<CaMP_run_folder>          \
        --cap_dir    <path>/<CaP_run_folder>           \
        --dense_dir  <path>/<dense_run_folder>         \   # optional
        --oracle_root <path>/<oracle_runs_root>        \   # optional
        --topology Dtelekom                            \
        --n_draws 10                                   \
        --pr 0.75                                      \
        --out_dir sensitivity_out                      \
        --device cpu

The script writes:
    sensitivity_out/sensitivity_<topology>.csv
    sensitivity_out/sensitivity_<topology>_lognormal.tex
    sensitivity_out/sensitivity_<topology>_reshuffle.tex
    sensitivity_out/sensitivity_<topology>_spike.tex

Oracle folder layout (when provided)
-------------------------------------
    <oracle_root>/<perturbation_key>/<draw_id>/{fine_tuned.pt, partition_final}
where perturbation_key is e.g. "lognormal_sigma0.5" and draw_id is "00".."09".
A separate script (to be written) will generate these oracle runs.
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from source import models
from source.utils.misc import get_layers, get_bn_layers

from pareto import (
    load_partition_data,
    compute_communication_cost_and_kbits,
    compute_communication_loss,
)


# ── perturbation families ────────────────────────────────────────────────────

def perturb_lognormal(maps, sigma, rng):
    """C'[i,j] = C[i,j] * exp(N(0, sigma^2))  for i != j.  Diagonal preserved."""
    C = np.asarray(maps, dtype=float).copy()
    n = C.shape[0]
    noise = rng.normal(0.0, sigma, size=(n, n))
    np.fill_diagonal(noise, 0.0)
    C = C * np.exp(noise)
    np.fill_diagonal(C, 0.0)
    return C.tolist()


def perturb_speed_class_reshuffle(maps, p, rng):
    """
    Approximate "redraw the speed class of fraction p of machines" by
    permuting their row+column blocks with another randomly chosen machine.

    The codebase does not store an explicit speed-class assignment per machine
    (the cost matrix is pre-baked into partition_final). This is the closest
    in-distribution approximation: each affected machine inherits another
    machine's connectivity profile, preserving the overall cost distribution
    while abruptly changing per-machine behaviour -- which is what a class
    change models.
    """
    C = np.asarray(maps, dtype=float).copy()
    n = C.shape[0]
    n_reshuffle = max(1, int(round(p * n)))
    affected = rng.choice(n, size=n_reshuffle, replace=False)
    donors = rng.integers(0, n, size=n_reshuffle)
    new_C = C.copy()
    for m, donor in zip(affected, donors):
        if donor == m:
            continue
        new_C[m, :] = C[donor, :]
        new_C[:, m] = C[:, donor]
    np.fill_diagonal(new_C, 0.0)
    return new_C.tolist()


def perturb_single_link_spike(maps, k, rng):
    """Pick one random off-diagonal entry (i,j) and multiply C[i,j] *and*
    C[j,i] by k. Captures a localized congestion event on a symmetric link."""
    C = np.asarray(maps, dtype=float).copy()
    n = C.shape[0]
    off_diag = [(i, j) for i in range(n) for j in range(n) if i != j]
    idx = rng.integers(0, len(off_diag))
    i, j = off_diag[idx]
    C[i, j] *= k
    C[j, i] *= k
    return C.tolist()


PERTURBATIONS = {
    "lognormal":  {"levels": [0.1, 0.25, 0.5, 1.0], "param_name": "sigma",
                   "fn": perturb_lognormal},
    "reshuffle":  {"levels": [0.1, 0.25, 0.5],       "param_name": "p",
                   "fn": perturb_speed_class_reshuffle},
    "spike":      {"levels": [2, 5, 10],              "param_name": "k",
                   "fn": perturb_single_link_spike},
}


# ── model loading (mirrors comm_cost_variance_sweep.load_experiment) ─────────

def build_model(model_name, num_classes, device):
    if model_name == "resnet18":
        m = models.__dict__[model_name](nn.Conv2d, nn.BatchNorm2d,
                                        num_classes=num_classes)
    elif model_name == "resnet101":
        m = models.__dict__[model_name](get_layers("regular"),
                                        get_bn_layers("regular"),
                                        num_classes=num_classes)
    else:
        m = models.__dict__[model_name](num_classes=num_classes)
    return m.to(device)


def load_run(folder, model_name, num_classes, device):
    """Load (model, partition) from a single experiment folder."""
    state_path = os.path.join(folder, "fine_tuned.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"No fine_tuned.pt in {folder}")
    partition = load_partition_data(folder)
    if partition is None:
        raise FileNotFoundError(f"No partition_final in {folder}")
    model = build_model(model_name, num_classes, device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    return model, partition


# ── analytic dense-frozen cost ───────────────────────────────────────────────

def dense_comm_cost(partition):
    """
    Realized comm cost for a fully-dense model: every (output filter, input
    channel) pair across machines requires communication, so every cross-
    machine edge is "lit" for every layer in partition.

    Equivalent to compute_communication_cost_and_kbits with all weights
    treated as nonzero. We re-implement here to avoid having to materialise
    a dense checkpoint.
    """
    if not partition or "num" not in partition or "maps" not in partition:
        return 0.0
    num_machines = partition["num"]
    C = partition["maps"]
    total = 0.0
    for layer_partition in partition.values():
        if not isinstance(layer_partition, dict):
            continue
        if "filter_id" not in layer_partition:
            continue
        parents = layer_partition.get("parents", [])
        if not parents:
            continue
        outsize = layer_partition.get("outsize", 1) or 1
        for n in range(num_machines):
            out_filters_here = len(layer_partition["filter_id"][n])
            if out_filters_here == 0:
                continue
            for i in range(num_machines):
                if i == n:
                    continue
                if len(parents) == 1:
                    in_chans = partition[parents[0]]["filter_id"][i]
                else:
                    in_chans = np.concatenate(
                        [partition[p]["filter_id"][i] for p in parents])
                    in_chans = np.unique(in_chans)
                if len(in_chans) > 0:
                    total += C[i][n] * outsize
    return total


# ── core sweep ───────────────────────────────────────────────────────────────

def eval_on_perturbed(model, partition, new_maps):
    """Return (pencr, penbr) under the perturbed cost matrix."""
    part = copy.deepcopy(partition)
    part["maps"] = new_maps
    pencr, _ = compute_communication_cost_and_kbits(model, part)
    penbr = compute_communication_loss(model, part)
    return float(pencr), float(penbr)


def try_load_oracle(oracle_root, ptype, param, draw_id, model_name,
                    num_classes, device):
    """Look for an oracle model trained under the perturbed cost C' for this
    specific (perturbation, parameter, draw). Returns (model, partition) or
    None."""
    if oracle_root is None:
        return None
    key = f"{ptype}_{PERTURBATIONS[ptype]['param_name']}{param}"
    folder = os.path.join(oracle_root, key, f"{int(draw_id):02d}")
    if not os.path.isdir(folder):
        return None
    try:
        return load_run(folder, model_name, num_classes, device)
    except FileNotFoundError:
        return None


def run_sensitivity(args):
    rng = np.random.default_rng(args.seed)

    camp_model, camp_part = load_run(args.camp_dir, args.model, args.num_classes,
                                     args.device)
    cap_model,  cap_part  = load_run(args.cap_dir,  args.model, args.num_classes,
                                     args.device)

    # The two methods must have been trained on the same physical topology /
    # cost matrix; we anchor perturbations to CaMP's partition['maps'].
    base_maps = np.asarray(camp_part["maps"], dtype=float)

    dense_part = None
    dense_model = None
    if args.dense_dir is not None:
        dense_model, dense_part = load_run(args.dense_dir, args.model,
                                           args.num_classes, args.device)

    rows = []

    for ptype, spec in PERTURBATIONS.items():
        fn = spec["fn"]
        for param in spec["levels"]:
            for draw in range(args.n_draws):
                new_maps = fn(base_maps.tolist(), param, rng)

                # CaMP frozen (both metrics)
                camp_pencr, camp_penbr = eval_on_perturbed(
                    camp_model, camp_part, new_maps)
                rows.append(dict(perturbation=ptype, param=param, draw=draw,
                                 method="CaMP-frozen", metric="pencr",
                                 value=camp_pencr))
                rows.append(dict(perturbation=ptype, param=param, draw=draw,
                                 method="CaMP-frozen", metric="penbr",
                                 value=camp_penbr))

                # CaP baseline frozen (exact metric only)
                cap_pencr, _ = eval_on_perturbed(cap_model, cap_part, new_maps)
                rows.append(dict(perturbation=ptype, param=param, draw=draw,
                                 method="CaP-frozen", metric="pencr",
                                 value=cap_pencr))

                # Dense frozen: real checkpoint if given, else analytic
                if dense_model is not None:
                    dense_pencr, _ = eval_on_perturbed(dense_model, dense_part,
                                                      new_maps)
                else:
                    # analytic dense doesn't depend on weights; it depends on
                    # the partition + C' only -- swap C' into a partition copy
                    part = copy.deepcopy(camp_part)
                    part["maps"] = new_maps
                    dense_pencr = dense_comm_cost(part)
                rows.append(dict(perturbation=ptype, param=param, draw=draw,
                                 method="Dense-frozen", metric="pencr",
                                 value=dense_pencr))

                # Oracle (CaMP retrained under C'): only if user provided dir
                oracle = try_load_oracle(args.oracle_root, ptype, param, draw,
                                         args.model, args.num_classes,
                                         args.device)
                if oracle is not None:
                    ora_model, ora_part = oracle
                    # The oracle was optimised *for* C'; evaluate on the same
                    # C'. ora_part['maps'] should already equal new_maps, but
                    # we re-set it for safety.
                    ora_pencr, _ = eval_on_perturbed(ora_model, ora_part,
                                                    new_maps)
                else:
                    ora_pencr = np.nan
                rows.append(dict(perturbation=ptype, param=param, draw=draw,
                                 method="CaMP-Cprime-oracle", metric="pencr",
                                 value=ora_pencr))

    df = pd.DataFrame(rows)
    df["topology"] = args.topology
    df["pr"] = args.pr
    return df


# ── output: CSV + LaTeX ──────────────────────────────────────────────────────

def emit_csv(df, out_dir, topology):
    path = os.path.join(out_dir, f"sensitivity_{topology}.csv")
    df.to_csv(path, index=False)
    print(f"  CSV  -> {path}")


def _agg(df, ptype, method, metric):
    """Mean across draws, indexed by perturbation parameter."""
    sub = df[(df.perturbation == ptype) &
             (df.method == method) &
             (df.metric == metric)]
    return sub.groupby("param")["value"].mean()


def _fmt(x):
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "--"
    if abs(x) >= 1e5:
        return f"{x:.2e}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.2f}"


PERT_HEADERS = {
    "lognormal": ("Realized Comm.~Cost under $\\C'$ ($\\sigma$ for log-normal noise)",
                  "$\\sigma$"),
    "reshuffle": ("Realized Comm.~Cost under $\\C'$ ($p$ for speed-class reshuffle)",
                  "$p$"),
    "spike":     ("Realized Comm.~Cost under $\\C'$ ($k$ for single-link spike)",
                  "$k$"),
}


def emit_latex(df, out_dir, topology, ptype, pr, n_draws):
    spec = PERTURBATIONS[ptype]
    levels = spec["levels"]
    header_top, header_unit = PERT_HEADERS[ptype]

    camp_penbr = _agg(df, ptype, "CaMP-frozen",          "penbr")
    camp_pencr = _agg(df, ptype, "CaMP-frozen",          "pencr")
    cap_pencr  = _agg(df, ptype, "CaP-frozen",           "pencr")
    dense      = _agg(df, ptype, "Dense-frozen",         "pencr")
    oracle     = _agg(df, ptype, "CaMP-Cprime-oracle",   "pencr")

    def row(label, series):
        cells = " & ".join(_fmt(series.get(lv, np.nan)) for lv in levels)
        return f"{label} & {cells}\\\\"

    n_cols = len(levels)
    col_spec = "l|" + "c" * n_cols
    header_cells = " & ".join(f"{header_unit}={lv}" for lv in levels)

    tex = f"""\\begin{{table}}[t]
\\centering
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{{col_spec}}}
\\toprule
\\textbf{{Method}} & \\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{header_top}}}}}\\\\
\\cline{{2-{n_cols + 1}}}
& {header_cells}\\\\
\\midrule
{row("\\Problemname-frozen ($\\penbr$)", camp_penbr)}
{row("\\Problemname-frozen ($\\pencr$)", camp_pencr)}
{row("\\Baseline-frozen",                 cap_pencr)}
{row("\\method{Dense}-frozen",            dense)}
\\midrule
{row("\\Problemname-$\\C'$ \\emph{(oracle)}", oracle)}
\\bottomrule
\\end{{tabular}}
}}
\\caption{{Realized communication cost under {ptype} perturbations of $\\C$
at deployment, for the \\emph{{{topology}}} topology with $\\text{{pr}}={pr}$.
Values averaged over {n_draws} perturbation draws.}}
\\label{{tab:sensitivity_{topology}_{ptype}}}
\\end{{table}}
"""
    path = os.path.join(out_dir, f"sensitivity_{topology}_{ptype}.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"  LaTeX -> {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camp_dir",     required=True,
                   help="Trained CaMP experiment folder (fine_tuned.pt + partition_final).")
    p.add_argument("--cap_dir",      required=True,
                   help="Trained CaP baseline experiment folder.")
    p.add_argument("--dense_dir",    default=None,
                   help="Optional dense-model folder. If omitted, dense cost "
                        "is computed analytically from the partition.")
    p.add_argument("--oracle_root",  default=None,
                   help="Optional root holding oracle runs trained under each "
                        "perturbed C'. Layout: <root>/<ptype>_<param><val>/<draw>/.")
    p.add_argument("--topology",     required=True,
                   help="Topology name for output labelling (e.g. Dtelekom).")
    p.add_argument("--pr",           type=float, default=0.75)
    p.add_argument("--n_draws",      type=int,   default=10)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default="cpu")
    p.add_argument("--model",        default="resnet101",
                   help="Model arch as registered in source.models.")
    p.add_argument("--num_classes",  type=int,   default=100)
    p.add_argument("--out_dir",      default="sensitivity_out")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = run_sensitivity(args)
    emit_csv(df, args.out_dir, args.topology)
    for ptype in PERTURBATIONS:
        emit_latex(df, args.out_dir, args.topology, ptype, args.pr, args.n_draws)


if __name__ == "__main__":
    main()
