r"""
sensitivity_analysis.py
=======================

Fills in the Table from `dynamic-1-sensitivity.tex` (Sec. "Robustness to
Communication-Cost Variation") of the J_CaMP paper.

What the table reports
----------------------
For each perturbation family, the *realized* communication cost (\penc with
E^hrd) when the deployment-time cost matrix C is replaced by a perturbed
matrix C'. Five rows in the draft table:

    - CaMP-frozen (\penbr) = CaMP trained with the TCC penalty, evaluated on C'
    - CaMP-frozen (\pencr) = CaMP trained with the AOP penalty, evaluated on C'
    - CaP-frozen           = baseline trained for C, evaluated on C'
    - Dense-frozen         = no pruning, evaluated on C'
    - CaMP-C' oracle       = full ADMM re-run from scratch with C' as input

The two CaMP-frozen rows are different *models* (TCC- and AOP-trained per
J_CaMP Sec. 5), both evaluated with the same realized-cost metric on C'.

Three perturbation families (each gets one table; baseline column added):

    1. Multiplicative log-normal noise    σ ∈ {0, 0.1, 0.25, 0.5, 1.0}
       (σ=0 ⇒ C'=C, baseline / sanity check)
    2. Speed-class reshuffle              p ∈ {0, 0.1, 0.25, 0.5}
    3. Single-link spike                   k ∈ {1, 2, 5, 10}

Why training is also perturbed (the oracle)
-------------------------------------------
Per the paper (Sec. 4-methodology and Alg.~ADMM), training takes C as input
to the penalty term and produces both weights W* and a placement M*. The
oracle row is therefore not a re-evaluation of the same model under C', but
a *new* model produced by re-running the full optimization with C'
substituted everywhere C appears -- inside the SGD primary update, inside
the greedy assignment subproblem, etc. This script wires that retraining
through `MoP.prune()` + `MoP.finetune()`.

Modes
-----
    eval          : only evaluate (CaMP-frozen, CaP-frozen, Dense-frozen).
                    Oracles loaded from disk if --oracle_root points at
                    completed retrains; missing oracle cells become "--".
                    Cheap. Default.
    train_oracles : for each (ptype, level, draw), retrain CaMP from scratch
                    with the perturbed C' as input, save to
                    <oracle_root>/<ptype>_<param><val>/<draw>/. Expensive.
                    Requires --config (the base training YAML).
    all           : train_oracles, then eval.

The eval pass and the train_oracles pass use the SAME C' for the same
(ptype, level, draw) tuple, via a deterministic per-cell seed derived from
--seed. Reproducible across reruns.

Usage
-----
    # cheap: just make the table from already-trained artifacts
    python sensitivity_analysis.py \
        --mode eval                                       \
        --camp_tcc_dir <path>/<CaMP_TCC_run_folder>       \
        --camp_aop_dir <path>/<CaMP_AOP_run_folder>       \
        --cap_dir      <path>/<CaP_run_folder>            \
        --topology Dtelekom                               \
        --pr 0.75 --n_draws 10                            \
        --oracle_root  oracle_runs/Dtelekom_pr0.75        \   # optional
        --out_dir sensitivity_out

    # expensive: actually train the oracles in-process
    python sensitivity_analysis.py \
        --mode train_oracles                              \
        --config config/resnet101-np11-Abilene_cost.yaml  \
        --camp_tcc_dir <path>/<CaMP_TCC_run_folder>       \
        --topology Abilene --pr 0.75 --n_draws 10        \
        --oracle_root  oracle_runs/Abilene_pr0.75
"""

import argparse
import copy
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

# Make CaP2 importable when this script is run from anywhere
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from source import models
from source.utils.misc import get_layers, get_bn_layers

from pareto import (
    load_partition_data,
    compute_communication_cost_and_kbits,
)


# ── perturbation families ────────────────────────────────────────────────────

def perturb_lognormal(maps, sigma, rng):
    """C'[i,j] = C[i,j] * exp(N(0, sigma^2)) for i != j. σ=0 ⇒ identity."""
    if sigma == 0:
        return [list(row) for row in maps]
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
    p=0 ⇒ identity.

    The codebase does not store an explicit speed-class assignment per
    machine (it's used transiently to construct C and then discarded -- see
    J_CaMP paper Sec. 5, line ~113). This row+column swap is the closest
    in-distribution approximation: each affected machine inherits another
    machine's connectivity profile.
    """
    if p == 0:
        return [list(row) for row in maps]
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
    """Pick one random off-diagonal entry (i,j), multiply C[i,j] *and*
    C[j,i] by k. k=1 ⇒ identity."""
    if k == 1:
        return [list(row) for row in maps]
    C = np.asarray(maps, dtype=float).copy()
    n = C.shape[0]
    off_diag = [(i, j) for i in range(n) for j in range(n) if i != j]
    idx = int(rng.integers(0, len(off_diag)))
    i, j = off_diag[idx]
    C[i, j] *= k
    C[j, i] *= k
    return C.tolist()


PERTURBATIONS = {
    "lognormal":  {"levels": [0.0, 0.1, 0.25, 0.5, 1.0],
                   "param_name": "sigma",
                   "baseline":   0.0,
                   "fn": perturb_lognormal},
    "reshuffle":  {"levels": [0.0, 0.1, 0.25, 0.5],
                   "param_name": "p",
                   "baseline":   0.0,
                   "fn": perturb_speed_class_reshuffle},
    "spike":      {"levels": [1, 2, 5, 10],
                   "param_name": "k",
                   "baseline":   1,
                   "fn": perturb_single_link_spike},
}


def derive_seed(ptype, level, draw, master_seed):
    """Deterministic seed for the (ptype, level, draw) cell. Used by both
    eval and train_oracles so the oracle is trained on the exact C' the
    frozen models get evaluated on."""
    key = f"{ptype}|{level}|{draw}|{master_seed}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def sample_C_prime(ptype, level, draw, base_maps, master_seed):
    rng = np.random.default_rng(derive_seed(ptype, level, draw, master_seed))
    return PERTURBATIONS[ptype]["fn"](base_maps.tolist(), level, rng)


def is_baseline(ptype, level):
    return level == PERTURBATIONS[ptype]["baseline"]


def perturbation_key(ptype, level):
    pname = PERTURBATIONS[ptype]["param_name"]
    return f"{ptype}_{pname}{level}"


# ── model loading ────────────────────────────────────────────────────────────

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
    """Realized comm cost for a fully-dense model: every (output filter,
    input channel) pair across machines requires communication. Avoids
    needing a dense checkpoint on disk."""
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
            if len(layer_partition["filter_id"][n]) == 0:
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


# ── evaluation ───────────────────────────────────────────────────────────────

def eval_on_perturbed(model, partition, new_maps):
    """Return the realized communication cost under the perturbed cost matrix
    (paper's eval metric: \\penc with E^hrd, computed by
    compute_communication_cost_and_kbits)."""
    part = copy.deepcopy(partition)
    part["maps"] = new_maps
    cost, _ = compute_communication_cost_and_kbits(model, part)
    return float(cost)


def oracle_save_dir(oracle_root, ptype, level, draw):
    if oracle_root is None:
        return None
    return os.path.join(oracle_root, perturbation_key(ptype, level),
                        f"{int(draw):02d}")


def try_load_oracle(oracle_root, ptype, level, draw, model_name,
                    num_classes, device):
    """Return (model, partition) if the oracle for this cell exists, else None."""
    folder = oracle_save_dir(oracle_root, ptype, level, draw)
    if folder is None or not os.path.isdir(folder):
        return None
    try:
        return load_run(folder, model_name, num_classes, device)
    except FileNotFoundError:
        return None


def run_eval(args):
    """Evaluate all rows on the perturbed cost matrices and build a tidy DataFrame.

    Two CaMP models are loaded -- one trained with the TCC penalty
    (\\penbr family) and one trained with the AOP penalty (\\pencr family),
    matching the two \\Problemname-frozen rows in the draft table. Both are
    evaluated under the same realized-cost metric on each C'.
    """
    camp_tcc_model, camp_tcc_part = load_run(args.camp_tcc_dir, args.model,
                                             args.num_classes, args.device)
    camp_aop_model, camp_aop_part = load_run(args.camp_aop_dir, args.model,
                                             args.num_classes, args.device)
    cap_model, cap_part = load_run(args.cap_dir, args.model,
                                   args.num_classes, args.device)

    # Both CaMP variants are trained on the same physical topology, so their
    # partition_final.maps must agree. We anchor perturbations to the TCC
    # variant by convention; sanity-check the AOP variant agrees.
    base_maps = np.asarray(camp_tcc_part["maps"], dtype=float)
    aop_maps  = np.asarray(camp_aop_part["maps"], dtype=float)
    if base_maps.shape != aop_maps.shape or not np.allclose(base_maps, aop_maps):
        print("WARNING: --camp_tcc_dir and --camp_aop_dir have different "
              "partition['maps']. They must be trained on the same topology. "
              "Anchoring perturbations to the TCC variant's cost matrix.")

    dense_model = None
    dense_part = None
    if args.dense_dir is not None:
        dense_model, dense_part = load_run(args.dense_dir, args.model,
                                           args.num_classes, args.device)

    rows = []
    for ptype, spec in PERTURBATIONS.items():
        for level in spec["levels"]:
            for draw in range(args.n_draws):
                new_maps = sample_C_prime(ptype, level, draw, base_maps,
                                          args.seed)

                # Two CaMP-frozen rows: one per training penalty.
                tcc_val = eval_on_perturbed(camp_tcc_model, camp_tcc_part,
                                            new_maps)
                aop_val = eval_on_perturbed(camp_aop_model, camp_aop_part,
                                            new_maps)
                rows.append(dict(perturbation=ptype, param=level, draw=draw,
                                 method="CaMP-frozen-TCC", value=tcc_val))
                rows.append(dict(perturbation=ptype, param=level, draw=draw,
                                 method="CaMP-frozen-AOP", value=aop_val))

                # CaP baseline frozen
                cap_val = eval_on_perturbed(cap_model, cap_part, new_maps)
                rows.append(dict(perturbation=ptype, param=level, draw=draw,
                                 method="CaP-frozen", value=cap_val))

                # Dense: real checkpoint if given, else analytic
                if dense_model is not None:
                    dense_val = eval_on_perturbed(dense_model, dense_part,
                                                  new_maps)
                else:
                    part = copy.deepcopy(camp_tcc_part)
                    part["maps"] = new_maps
                    dense_val = dense_comm_cost(part)
                rows.append(dict(perturbation=ptype, param=level, draw=draw,
                                 method="Dense-frozen", value=dense_val))

                # Oracle: skipped entirely if --oracle_root not provided.
                # Baseline level (C'=C) reuses the CaMP-TCC value since
                # retraining under C'=C is the original training.
                if args.oracle_root is not None:
                    if is_baseline(ptype, level):
                        ora_val = tcc_val
                    else:
                        oracle = try_load_oracle(args.oracle_root, ptype,
                                                 level, draw, args.model,
                                                 args.num_classes, args.device)
                        if oracle is None:
                            ora_val = float("nan")
                        else:
                            ora_model, ora_part = oracle
                            ora_val = eval_on_perturbed(ora_model, ora_part,
                                                        new_maps)
                    rows.append(dict(perturbation=ptype, param=level,
                                     draw=draw,
                                     method="CaMP-Cprime-oracle",
                                     value=ora_val))

    df = pd.DataFrame(rows)
    df["topology"] = args.topology
    df["pr"] = args.pr
    return df


# ── oracle training ──────────────────────────────────────────────────────────

def _write_perturbed_partition_yaml(base_partition_yaml, new_maps, out_path):
    """Copy the base partition YAML and replace its 'maps' field with new_maps.
    Returns out_path. Used to feed C' into the existing MoP/generate_partition
    pipeline without monkey-patching it."""
    with open(base_partition_yaml, "r") as f:
        d = yaml.safe_load(f)
    d["maps"] = [[float(x) for x in row] for row in new_maps]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(d, f)
    return out_path


def _load_training_config(config_yaml):
    with open(config_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_one_oracle(base_config, base_partition_yaml, new_maps, save_dir,
                     work_dir):
    """Re-run the full CaMP optimization with C' substituted for C.

    base_config        : dict loaded from the original training YAML
    base_partition_yaml: path to the original config/<topology>.yaml used at
                         training time (the one whose `maps` we replace)
    new_maps           : list-of-list, the perturbed C'
    save_dir           : where the oracle artifacts (fine_tuned.pt,
                         partition_final, best_accuracy.txt) should land
    work_dir           : where to drop the temporary perturbed-partition YAML
    """
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(os.path.join(save_dir, "fine_tuned.pt")):
        print(f"  [skip] oracle already trained at {save_dir}")
        return save_dir

    perturbed_yaml = os.path.join(
        work_dir, f"perturbed_partition_{int(time.time()*1000)}.yaml")
    _write_perturbed_partition_yaml(base_partition_yaml, new_maps,
                                    perturbed_yaml)

    cfg = copy.deepcopy(base_config)
    cfg["partition_path"] = perturbed_yaml
    cfg["experiment_dir"] = save_dir
    cfg["log_dir"]        = os.path.dirname(save_dir) or "."
    cfg["plot"]           = False
    cfg.setdefault("use_wandb", False)
    # The training pipeline references some keys that argparse defaults usually
    # supply; make sure they exist:
    cfg.setdefault("create_partition", False)
    cfg.setdefault("load_pruned_model", False)
    cfg.setdefault("load_dense_model",  cfg.get("load_dense_model", False))

    # Late import: only pull MoP when we actually need to train, so that pure
    # eval runs don't pay the heavy import cost.
    from source.core.engine import MoP

    print(f"  [train] {save_dir}")
    mop = MoP(cfg)
    mop.prune()
    mop.finetune()

    try:
        os.remove(perturbed_yaml)
    except OSError:
        pass
    return save_dir


def run_train_oracles(args):
    """Train one oracle per (ptype, level, draw), skipping baseline levels
    (C'=C ⇒ retraining is redundant) and any cells already on disk."""
    if args.config is None:
        raise SystemExit("--config (base training YAML) is required for "
                         "--mode train_oracles or all.")
    if args.oracle_root is None:
        raise SystemExit("--oracle_root is required for --mode train_oracles "
                         "or all.")

    base_config = _load_training_config(args.config)

    # Where does the base config expect to find the partition YAML?
    # run_partition.py builds it dynamically from {model, num_partition,
    # experiment_name}, but for our purposes the base config that the user
    # passes us already names a topology -- read its partition path the same
    # way.
    base_partition_yaml = base_config.get("partition_path") or os.path.join(
        _HERE, "config",
        f"{base_config['model']}-np{base_config['num_partition']}-"
        f"{base_config['experiment_name']}.yaml")
    if not os.path.exists(base_partition_yaml):
        raise FileNotFoundError(
            f"Cannot find base partition YAML at {base_partition_yaml}. "
            "Pass --base_partition_yaml to override.")

    work_dir = os.path.join(args.out_dir, "_perturbed_yamls")
    os.makedirs(work_dir, exist_ok=True)

    # We anchor perturbations to the TCC-trained CaMP run's cost matrix.
    # (Both CaMP variants share the same physical topology, so this is just a
    # choice of anchor.)
    _, camp_part = load_run(args.camp_tcc_dir, args.model, args.num_classes,
                            args.device)
    base_maps = np.asarray(camp_part["maps"], dtype=float)

    n_planned = 0
    n_done    = 0
    for ptype, spec in PERTURBATIONS.items():
        for level in spec["levels"]:
            if is_baseline(ptype, level):
                continue
            for draw in range(args.n_draws):
                n_planned += 1
                save_dir = oracle_save_dir(args.oracle_root, ptype, level, draw)
                new_maps = sample_C_prime(ptype, level, draw, base_maps,
                                          args.seed)
                try:
                    train_one_oracle(base_config, base_partition_yaml,
                                     new_maps, save_dir, work_dir)
                    n_done += 1
                except Exception as e:  # don't let one failure halt the sweep
                    print(f"  [error] training {save_dir} failed: {e}")
    print(f"\nOracle training complete: {n_done}/{n_planned} runs succeeded.")


# ── output: CSV + LaTeX ──────────────────────────────────────────────────────

def emit_csv(df, out_dir, topology):
    path = os.path.join(out_dir, f"sensitivity_{topology}.csv")
    df.to_csv(path, index=False)
    print(f"  CSV   -> {path}")


def _agg(df, ptype, method):
    sub = df[(df.perturbation == ptype) & (df.method == method)]
    return sub.groupby("param")["value"].mean()


def _fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
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

    camp_tcc = _agg(df, ptype, "CaMP-frozen-TCC")
    camp_aop = _agg(df, ptype, "CaMP-frozen-AOP")
    cap      = _agg(df, ptype, "CaP-frozen")
    dense    = _agg(df, ptype, "Dense-frozen")
    has_oracle = (df["method"] == "CaMP-Cprime-oracle").any()
    oracle   = _agg(df, ptype, "CaMP-Cprime-oracle") if has_oracle else None

    def row(label, series):
        cells = " & ".join(_fmt(series.get(lv, np.nan)) for lv in levels)
        return f"{label} & {cells}\\\\"

    n_cols = len(levels)
    col_spec = "l|" + "c" * n_cols
    header_cells = " & ".join(f"{header_unit}={lv}" for lv in levels)

    body_lines = [
        row("\\Problemname-frozen ($\\penbr$)", camp_tcc),
        row("\\Problemname-frozen ($\\pencr$)", camp_aop),
        row("\\Baseline-frozen",                cap),
        row("\\method{Dense}-frozen",           dense),
    ]
    if oracle is not None:
        body_lines.append("\\midrule")
        body_lines.append(row("\\Problemname-$\\C'$ \\emph{(oracle)}", oracle))
    body = "\n".join(body_lines)

    tex = f"""\\begin{{table}}[t]
\\centering
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{{col_spec}}}
\\toprule
\\textbf{{Method}} & \\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{header_top}}}}}\\\\
\\cline{{2-{n_cols + 1}}}
& {header_cells}\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\caption{{Realized communication cost under {ptype} perturbations of $\\C$
at deployment, for the \\emph{{{topology}}} topology with $\\text{{pr}}={pr}$.
The leftmost column ({header_unit}={spec['baseline']}) is the no-perturbation
baseline ($\\C'=\\C$). Values averaged over {n_draws} perturbation draws.}}
\\label{{tab:sensitivity_{topology}_{ptype}}}
\\end{{table}}
"""
    path = os.path.join(out_dir, f"sensitivity_{topology}_{ptype}.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"  LaTeX -> {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["eval", "train_oracles", "all"],
                   default="eval")

    p.add_argument("--camp_tcc_dir", required=True,
                   help="Trained CaMP-TCC experiment folder (model trained "
                        "with the \\penbr / TCC penalty). Used for the first "
                        "\\Problemname-frozen row.")
    p.add_argument("--camp_aop_dir", default=None,
                   help="Trained CaMP-AOP experiment folder (model trained "
                        "with the \\pencr / AOP penalty). Used for the second "
                        "\\Problemname-frozen row. Required for --mode eval/all.")
    p.add_argument("--cap_dir",  default=None,
                   help="Trained CaP baseline folder. Required for --mode eval/all.")
    p.add_argument("--dense_dir", default=None,
                   help="Optional dense-model folder. If omitted, dense cost "
                        "is computed analytically from the partition.")

    p.add_argument("--oracle_root", default=None,
                   help="Root holding oracle runs trained under each C'. "
                        "Layout: <root>/<ptype>_<param><val>/<draw>/.")
    p.add_argument("--config", default=None,
                   help="Base training YAML (e.g. config/cifar10.yaml). "
                        "Required for --mode train_oracles/all.")
    p.add_argument("--base_partition_yaml", default=None,
                   help="Override the topology partition YAML path. Default "
                        "is inferred from the training config.")

    p.add_argument("--topology",  required=True,
                   help="Topology name for output labelling (e.g. Dtelekom).")
    p.add_argument("--pr",        type=float, default=0.75)
    p.add_argument("--n_draws",   type=int,   default=10)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--device",    default="cpu")
    p.add_argument("--model",     default="resnet101")
    p.add_argument("--num_classes", type=int, default=100)
    p.add_argument("--out_dir",   default="sensitivity_out")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode in ("train_oracles", "all"):
        if args.base_partition_yaml is not None and args.config is not None:
            # Inject so run_train_oracles picks it up without recomputing
            cfg = _load_training_config(args.config)
            cfg["partition_path"] = args.base_partition_yaml
            tmp_cfg_path = os.path.join(args.out_dir, "_resolved_config.yaml")
            with open(tmp_cfg_path, "w") as f:
                yaml.safe_dump(cfg, f)
            args.config = tmp_cfg_path
        run_train_oracles(args)

    if args.mode in ("eval", "all"):
        missing = [name for name, val in
                   (("--camp_aop_dir", args.camp_aop_dir),
                    ("--cap_dir",      args.cap_dir)) if val is None]
        if missing:
            raise SystemExit(f"{', '.join(missing)} required for --mode eval/all.")
        df = run_eval(args)
        emit_csv(df, args.out_dir, args.topology)
        for ptype in PERTURBATIONS:
            emit_latex(df, args.out_dir, args.topology, ptype,
                       args.pr, args.n_draws)


if __name__ == "__main__":
    main()
