r"""
sensitivity_analysis.py
=======================

Generates the per-topology robustness plot for `dynamic-1-sensitivity.tex`
(Sec. "Robustness to Communication-Cost Variation") of the J_CaMP paper.

What the plots show
-------------------
Two plots per topology, one per dispersion statistic, each comparing
methods on a log-sigma axis. The MEAN relative growth is NOT plotted:
the debiased perturbation has E[cost(C')] = cost(C) by construction
(cost is linear in C'), so the empirical mean is an unbiased estimator
of *zero* growth and the only signal is sampling noise. The honest
robustness question is "how much does the realized cost VARY around its
training-time value?", which is what std and MAD measure. Files emitted
per topology T:

    sensitivity_T_std.pdf  : y = Std(cost(C')) / cost(C)        (relative std)
    sensitivity_T_mad.pdf  : y = E[|cost(C') - cost(C)|] / cost(C)  (relative MAD)

Both start at 0 at sigma=0 and grow monotonically. Curves compared:

    - CaMP-frozen (TCC penalty / \penbr)
    - CaMP-frozen (AOP penalty / \pencr)
    - CaP-frozen   (baseline)
    - Dense-frozen (no pruning)
    - CaMP-C' oracle  (optional, drawn if oracle artifacts are on disk)

Each method's curve has TWO renderings: empirical (solid line + markers,
from n_draws independent perturbations of C) and analytical (dashed
line, same color), where the analytical curve is closed-form for std
and fast iid Monte Carlo on the log-normal multipliers for MAD. The
analytical lines exist because cost(C') is linear in C' for a frozen
model: with X_(i,n) = (active comm. weight) * C[i,n],
    Var(cost(C')) = sigma^2 * sum X^2
    => Std/cost(C) = sigma * sqrt(sum X^2) / sum X,
and MAD has no closed form but can be Monte-Carloed cheaply from the
(active_weights, C, sigma) triple alone.

Perturbation model (debiased multiplicative log-normal, zero-mean noise)
------------------------------------------------------------------------
For each off-diagonal entry of C,

    C'[i,j] = C[i,j] * exp(eta - sigma_log^2 / 2),
    eta ~ N(0, sigma_log^2),  sigma_log = sqrt(log(1 + sigma^2))

is calibrated so that *exactly*:

    E[C'[i,j]]               = C[i,j]      (zero-mean noise)
    Std(C'[i,j]) / C[i,j]    = sigma       (target relative std)

C'[i,j] > 0 always, so no clipping or thresholding is needed. The other
two perturbation families (speed-class reshuffle, single-link spike)
have been dropped per the scope agreed for the appendix.

Modes
-----
    eval          : only evaluate the frozen models on each perturbed C'.
                    Oracles loaded from disk if --oracle_root points at
                    completed retrains; missing oracle cells become NaN.
                    Default; cheap.
    train_oracles : for each (sigma, draw), retrain CaMP from scratch
                    with C' as input, save to
                    <oracle_root>/lognormal_sigma<val>/<draw>/. Expensive.
                    Requires --config (the base training YAML).
    all           : train_oracles, then eval.

The eval and train_oracles passes use the SAME C' for the same
(sigma, draw) tuple, via a deterministic per-cell seed derived from
--seed. Reproducible across reruns.

Usage
-----
    # cheap: just make the plot from already-trained artifacts
    python sensitivity_analysis.py \
        --mode eval                                       \
        --camp_tcc_dir <path>/<CaMP_TCC_run_folder>       \
        --camp_aop_dir <path>/<CaMP_AOP_run_folder>       \
        --cap_dir      <path>/<CaP_run_folder>            \
        --topology Dtelekom                               \
        --pr 0.75 --n_draws 10                            \
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    """Debiased multiplicative log-normal perturbation calibrated so that
    sigma is the *relative standard deviation* of C':
        E[C'[i,j]]                = C[i,j]              (zero-mean noise)
        Std(C'[i,j]) / C[i,j]     = sigma               (target rel. std)

    Implemented as C'[i,j] = C[i,j] * exp(eta - sigma_log^2/2), with
    eta ~ N(0, sigma_log^2) and sigma_log = sqrt(log(1 + sigma^2)). C' is
    strictly positive (no clipping). Diagonal is zeroed. sigma=0 ⇒ identity.
    """
    if sigma == 0:
        return [list(row) for row in maps]
    C = np.asarray(maps, dtype=float).copy()
    n = C.shape[0]
    sigma_log = np.sqrt(np.log(1.0 + sigma ** 2))
    noise = rng.normal(0.0, sigma_log, size=(n, n))
    np.fill_diagonal(noise, 0.0)
    C = C * np.exp(noise - 0.5 * sigma_log ** 2)
    np.fill_diagonal(C, 0.0)
    return C.tolist()


# Dense sigma grid. Heavier near zero (the "graceful degradation" regime
# we care about) and progressively sparser at the tail. sigma=0 is the
# no-perturbation baseline. Combined with x-axis symlog this covers the
# small- and large-noise regions in one readable plot.
SIGMA_LEVELS = [0.0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75,
                1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]

PERTURBATIONS = {
    "lognormal":  {"levels":     SIGMA_LEVELS,
                   "param_name": "sigma",
                   "baseline":   0.0,
                   "fn":         perturb_lognormal},
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


# ── analytical relative-moment computation ───────────────────────────────────
# Because the realized cost is linear in C',
#
#   cost(C')  =  sum_(i,n)  w_(i,n) * C'_(i,n)
#
# with the per-pair weight w_(i,n) determined entirely by the frozen model
# and partition, the relative standard deviation and relative mean absolute
# deviation of cost(C') under the debiased zero-mean log-normal perturbation
# can be computed without any further frozen-model evaluations: only the
# (one-time) per-(i,n) weights and the cost matrix C are needed. The
# closed-form std follows from independence of L_(i,n); MAD has no closed
# form but is exact under fast iid Monte Carlo on the L_(i,n) multipliers.

def extract_active_weights_model(model, partition):
    """Per-(i,n) communication weight for a frozen model. Mirrors
    compute_communication_cost_and_kbits exactly, so
        sum_(i,n) weight[(i,n)] * C[i,n]  ==  cost(C)
    holds with whatever (model, partition) yielded the empirical cost.
    """
    weights = {}
    if not partition or "num" not in partition or "maps" not in partition:
        return weights
    num_machines = partition["num"]
    for name, W in model.named_parameters():
        if name not in partition:
            continue
        weight = W.cpu().detach().numpy()
        shape = weight.shape
        is_conv = (len(shape) == 4)
        layer_partition = partition[name]
        outsize = layer_partition.get("outsize", 1) or 1
        parents = layer_partition.get("parents", [])
        if not parents:
            continue
        if is_conv:
            W_flat = np.sum(np.abs(weight.reshape(shape[0], shape[1], -1)),
                            axis=2)
        else:
            W_flat = np.abs(weight)
        for n in range(num_machines):
            for C_out in layer_partition["filter_id"][n]:
                for i in range(num_machines):
                    if i == n:
                        continue
                    if len(parents) == 1:
                        input_channels = partition[parents[0]]["filter_id"][i]
                    else:
                        input_channels = np.concatenate(
                            [partition[p]["filter_id"][i] for p in parents])
                        input_channels = np.unique(input_channels)
                    if (len(input_channels) > 0
                            and np.any(W_flat[C_out, input_channels])):
                        weights[(i, n)] = weights.get((i, n), 0.0) + outsize
    return weights


def extract_active_weights_dense(partition):
    """Per-(i,n) weight matching the analytic dense_comm_cost formula.
    Used when no dense checkpoint is provided -- consistent with the
    empirical dense baseline so analytical and empirical lines overlap.
    """
    weights = {}
    if not partition or "num" not in partition or "maps" not in partition:
        return weights
    num_machines = partition["num"]
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
                    weights[(i, n)] = weights.get((i, n), 0.0) + outsize
    return weights


def analytical_relative_moments(active_weights, C_matrix, sigmas,
                                n_mc=20000, rng=None):
    """Return (rel_std_per_sigma, rel_mad_per_sigma) for the debiased
    zero-mean log-normal perturbation.

    Math: with X_(i,n) = weight[(i,n)] * C[i,n] and i.i.d. log-normal
    multipliers L_(i,n) of mean 1 and variance sigma^2,
        Var(cost(C')) = sigma^2 * sum X^2,
    giving the exact closed-form relative std
        std/cost(C) = sigma * sqrt(sum X^2) / sum X.
    Relative MAD has no closed form (weighted sum of log-normals) and is
    estimated via fast iid Monte Carlo on the L_(i,n) multipliers --
    this requires no frozen-model evaluations and so is "analytical" in
    the sense of depending only on (active_weights, C, sigma).
    """
    zero = {float(s): 0.0 for s in sigmas}
    if not active_weights:
        return zero, dict(zero)
    C_arr = np.asarray(C_matrix, dtype=float)
    keys = list(active_weights.keys())
    X = np.array([active_weights[k] * C_arr[k[0], k[1]] for k in keys],
                 dtype=float)
    cost_C = float(X.sum())
    if cost_C <= 0 or not np.isfinite(cost_C):
        return zero, dict(zero)
    sum_X2 = float((X * X).sum())
    rng = rng if rng is not None else np.random.default_rng(0)
    out_std, out_mad = {}, {}
    for sigma in sigmas:
        s = float(sigma)
        if s == 0.0:
            out_std[s] = 0.0
            out_mad[s] = 0.0
            continue
        out_std[s] = s * np.sqrt(sum_X2) / cost_C
        sigma_log = np.sqrt(np.log(1.0 + s ** 2))
        eta = rng.normal(0.0, sigma_log, size=(n_mc, len(X)))
        L = np.exp(eta - 0.5 * sigma_log ** 2)
        costs = L @ X
        out_mad[s] = float(np.mean(np.abs(costs - cost_C))) / cost_C
    return out_std, out_mad


# ── per-layer per-pair bits, for the delay experiment ───────────────────────
# The "realized cost" metric is linear in C and aggregates everything to a
# scalar. The delay experiment needs more structure: per layer, per
# (source machine, dest machine) pair, the bytes transmitted. With c_{i,n} =
# 1/bandwidth on link (i,n) (log-normal with mean C[i,n]), the per-pair
# delay on layer l is bytes_l[i,n] * c_{i,n}; the layer delay is the max
# over (i,n) (all pairs send in parallel); the pipelined and serial
# end-to-end delays are max_l and sum_l over those, respectively.

def compute_per_layer_per_pair_bits(model, partition):
    """Return a list of (num_machines x num_machines) bytes-transmitted
    matrices, one per layer that has parent layers. Matches the iteration
    structure of compute_communication_cost_and_kbits exactly:
    one outsize-byte message per (output filter on n, source machine i)
    pair, multiplied by 4 bytes/float. Diagonal entries are 0.
    """
    if not partition or "num" not in partition or "maps" not in partition:
        return [], []
    num_machines = partition["num"]
    layer_bits = []
    layer_names = []
    for name, W in model.named_parameters():
        if name not in partition:
            continue
        weight = W.cpu().detach().numpy()
        shape = weight.shape
        is_conv = (len(shape) == 4)
        layer_partition = partition[name]
        outsize = layer_partition.get("outsize", 1) or 1
        parents = layer_partition.get("parents", [])
        if not parents:
            continue
        if is_conv:
            W_flat = np.sum(np.abs(weight.reshape(shape[0], shape[1], -1)),
                            axis=2)
        else:
            W_flat = np.abs(weight)
        bits = np.zeros((num_machines, num_machines), dtype=float)
        for n in range(num_machines):
            for C_out in layer_partition["filter_id"][n]:
                for i in range(num_machines):
                    if i == n:
                        continue
                    if len(parents) == 1:
                        input_channels = partition[parents[0]]["filter_id"][i]
                    else:
                        input_channels = np.concatenate(
                            [partition[p]["filter_id"][i] for p in parents])
                        input_channels = np.unique(input_channels)
                    if (len(input_channels) > 0
                            and np.any(W_flat[C_out, input_channels])):
                        # One message of outsize floats (4 bytes each) per
                        # (output filter, source machine) pair.
                        bits[i, n] += outsize * 4.0
        layer_bits.append(bits)
        layer_names.append(name)
    return layer_bits, layer_names


def eval_delays_on_perturbed(layer_bits, new_maps):
    """Compute pipelined (max_l) and serial (sum_l) end-to-end delays
    under a perturbed cost matrix new_maps interpreted as c = 1/bandwidth.

    For each layer l, the layer delay d_l is the max over machine pairs of
    bytes_l[i,n] * c_{i,n}: all pairs transmit in parallel, so the slowest
    pair determines the layer's wall-clock. Then:
        pipelined = max_l d_l        (steady-state throughput-limited delay)
        serial    = sum_l d_l        (single-sample end-to-end latency)
    """
    if not layer_bits:
        return 0.0, 0.0
    C = np.asarray(new_maps, dtype=float)
    layer_d = [(b * C).max() for b in layer_bits]
    pipelined = float(max(layer_d)) if layer_d else 0.0
    serial = float(sum(layer_d))
    return pipelined, serial


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

    Returns (df, active_weights, base_maps), where active_weights[method]
    is the per-(i,n) communication-weight dict that lets
    analytical_relative_moments compute the relative std and MAD of
    cost(C') without further model evaluations.
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

    # Per-method active-weight extraction. Cheap, done once. Used by
    # analytical_relative_moments to overlay closed-form / MC curves on
    # the plot. For each method the equality
    #   sum_(i,n) weight[(i,n)] * C[i,n]  ==  cost(C)   (at sigma=0)
    # holds by construction (same loop structure as the empirical eval).
    active_weights = {
        "CaMP-frozen-TCC": extract_active_weights_model(
            camp_tcc_model, camp_tcc_part),
        "CaMP-frozen-AOP": extract_active_weights_model(
            camp_aop_model, camp_aop_part),
        "CaP-frozen":      extract_active_weights_model(
            cap_model, cap_part),
    }
    if dense_model is not None:
        active_weights["Dense-frozen"] = extract_active_weights_model(
            dense_model, dense_part)
    else:
        active_weights["Dense-frozen"] = extract_active_weights_dense(
            camp_tcc_part)

    rows = []
    for ptype in args.perturbations:
        spec = PERTURBATIONS[ptype]
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
    return df, active_weights, base_maps


def run_delays(args):
    """Compute pipelined and serial inference delays for each method under
    perturbations of 1/bandwidth.

    Model: for every link (i, n) the per-byte transmission cost
    c_{i,n} = 1/bandwidth_{i,n} is treated as log-normal with mean
    C[i,n] (the training-time cost matrix) and relative standard deviation
    sigma. For each layer l, the layer delay is
        d_l = max over (i,n) of bytes_l[i,n] * c_{i,n}
    (all machine pairs transmit in parallel). The two reported metrics
    aggregate across layers:
        pipelined = max_l d_l  (steady-state throughput-limited delay)
        serial    = sum_l d_l  (single-sample end-to-end latency)

    Returns a tidy DataFrame with one row per
    (perturbation, sigma, draw, method, metric) and column 'value'.
    """
    camp_tcc_model, camp_tcc_part = load_run(args.camp_tcc_dir, args.model,
                                             args.num_classes, args.device)
    camp_aop_model, camp_aop_part = load_run(args.camp_aop_dir, args.model,
                                             args.num_classes, args.device)
    cap_model, cap_part = load_run(args.cap_dir, args.model,
                                   args.num_classes, args.device)

    base_maps = np.asarray(camp_tcc_part["maps"], dtype=float)
    aop_maps  = np.asarray(camp_aop_part["maps"], dtype=float)
    if base_maps.shape != aop_maps.shape or not np.allclose(base_maps, aop_maps):
        print("WARNING: --camp_tcc_dir and --camp_aop_dir have different "
              "partition['maps']. Anchoring perturbations to the TCC variant.")

    # Per-(layer, i, n) bytes depend only on (model, partition), not on C.
    # Compute once per method; reuse across all draws.
    print("  Extracting per-layer per-pair bytes for each method ...")
    bits_tcc, _ = compute_per_layer_per_pair_bits(camp_tcc_model, camp_tcc_part)
    bits_aop, _ = compute_per_layer_per_pair_bits(camp_aop_model, camp_aop_part)
    bits_cap, _ = compute_per_layer_per_pair_bits(cap_model,      cap_part)

    method_bits = [
        ("CaMP-frozen-TCC", bits_tcc),
        ("CaMP-frozen-AOP", bits_aop),
        ("CaP-frozen",      bits_cap),
    ]

    rows = []
    for ptype in args.perturbations:
        spec = PERTURBATIONS[ptype]
        for level in spec["levels"]:
            for draw in range(args.n_draws):
                new_maps = sample_C_prime(ptype, level, draw, base_maps,
                                          args.seed)
                for method_name, bits in method_bits:
                    pipelined, serial = eval_delays_on_perturbed(bits, new_maps)
                    rows.append(dict(perturbation=ptype, param=level, draw=draw,
                                     method=method_name, metric="pipelined",
                                     value=pipelined))
                    rows.append(dict(perturbation=ptype, param=level, draw=draw,
                                     method=method_name, metric="serial",
                                     value=serial))

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
    for ptype in args.perturbations:
        spec = PERTURBATIONS[ptype]
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


# ── output: CSV + per-topology plot ──────────────────────────────────────────

def emit_csv(df, out_dir, topology):
    path = os.path.join(out_dir, f"sensitivity_{topology}.csv")
    df.to_csv(path, index=False)
    print(f"  CSV   -> {path}")


def _agg(df, ptype, method):
    """Aggregate mean/std/count over draws for one (ptype, method)."""
    sub = df[(df.perturbation == ptype) & (df.method == method)]
    return sub.groupby("param")["value"].agg(["mean", "std", "count"])


# Curves to draw, in the order they appear in the legend. Tuple is
# (method-name-in-df, legend-label, color, marker).
PLOT_METHODS = [
    ("CaMP-frozen-TCC",    r"\textsc{CaMP}-frozen (TCC)",      "tab:blue",   "o"),
    ("CaMP-frozen-AOP",    r"\textsc{CaMP}-frozen (AOP)",      "tab:orange", "s"),
    ("CaP-frozen",         r"\textsc{CaP}-frozen (baseline)",  "tab:green",  "^"),
    ("Dense-frozen",       r"Dense-frozen",                    "tab:red",    "D"),
    ("CaMP-Cprime-oracle", r"\textsc{CaMP}-$C'$ (oracle)",     "tab:purple", "v"),
]


def _empirical_stat(df_sub, statistic, baseline):
    """Return (sigmas, values) of the per-sigma empirical statistic.
    For "std" and "mad" the value is normalized by the (deterministic)
    sigma=0 baseline cost; for "mean" the absolute mean is returned
    (no normalization), so the y-axis shows E[cost(C')] in the native
    units of cost(C). Centering on cost(C) (rather than the empirical
    mean) is correct because E[cost(C')] = cost(C) holds exactly under
    our debiased noise.
    """
    if statistic == "std":
        agg = df_sub.groupby("param")["value"].agg(["std", "count"]).sort_index()
        y = agg["std"].fillna(0.0).to_numpy(dtype=float) / baseline
    elif statistic == "mad":
        sub2 = df_sub.copy()
        sub2["absdev"] = (sub2["value"] - baseline).abs()
        agg = sub2.groupby("param")["absdev"].agg(["mean", "count"]).sort_index()
        y = agg["mean"].to_numpy(dtype=float) / baseline
    elif statistic == "mean":
        agg = df_sub.groupby("param")["value"].agg(["mean", "count"]).sort_index()
        y = agg["mean"].to_numpy(dtype=float)
    else:
        raise ValueError(f"unknown statistic: {statistic}")
    return agg.index.to_numpy(dtype=float), y


STAT_LABELS = {
    "std": (r"$\mathrm{Std}\bigl(\mathrm{cost}(C')\bigr) / \mathrm{cost}(C)$",
            "relative std"),
    "mad": (r"$\mathbb{E}\bigl[\,|\mathrm{cost}(C')-\mathrm{cost}(C)|\,\bigr] / \mathrm{cost}(C)$",
            "relative MAD"),
    "mean": (r"$\mathbb{E}\bigl[\mathrm{cost}(C')\bigr]$",
             "absolute mean cost"),
}


def emit_plot(df, out_dir, topology, ptype, pr, n_draws, statistic,
              analyticals):
    """One PDF + PNG per topology + statistic. statistic in {'std', 'mad', 'mean'}.

    For 'std' and 'mad': plots the relative dispersion of realized cost
    under the debiased zero-mean log-normal perturbation, normalized by
    cost(C). For 'mean': plots the absolute E[cost(C')] in the native
    units of cost(C). By construction E[cost(C')] = cost(C), so the
    'mean' lines are approximately horizontal (up to MC noise) at each
    method's cost(C); their relative levels show that the per-method
    advantage CaMP < CaP persists for every sigma in expectation.

    Solid + markers = empirical (over n_draws perturbation samples).
    Dashed = analytical -- closed-form for std (exact) and iid Monte
    Carlo on the log-normal multipliers for MAD (no model eval); not
    drawn for 'mean' (the analytical is trivially the constant cost(C)).
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    sub_pt = df[df["perturbation"] == ptype]
    drawn = 0
    for method_key, label, color, marker in PLOT_METHODS:
        sub = sub_pt[sub_pt["method"] == method_key]
        if sub.empty or 0.0 not in sub["param"].values:
            continue
        baseline = float(sub[sub["param"] == 0.0]["value"].mean())
        if not np.isfinite(baseline) or baseline <= 0:
            continue
        sigmas_emp, y_emp = _empirical_stat(sub, statistic, baseline)
        ax.plot(sigmas_emp, y_emp, color=color, marker=marker, markersize=4.5,
                linewidth=1.6, label=label)
        an = analyticals.get(method_key)
        if an:
            an_sigmas = np.array(sorted(an.keys()), dtype=float)
            an_y = np.array([an[s] for s in an_sigmas], dtype=float)
            ax.plot(an_sigmas, an_y, color=color, linestyle="--",
                    linewidth=1.2, alpha=0.85)
        drawn += 1

    if drawn == 0:
        plt.close(fig)
        print(f"  Plot  -> (skipped: no data for {topology}/{statistic})")
        return

    ax.set_xscale("symlog", linthresh=0.05)
    if statistic == "mean":
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=0.05)
        ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel(r"Noise scale $\sigma$ (relative std of $C'$)")
    ylabel, short = STAT_LABELS[statistic]
    ax.set_ylabel(ylabel)
    ax.set_title(f"{topology}  (pr={pr}, {n_draws} draws/level)")
    ax.grid(True, which="both", alpha=0.3)

    # Two-column legend: method colors on the left, line-style key on the right.
    from matplotlib.lines import Line2D
    method_h, method_l = ax.get_legend_handles_labels()
    style_h = [
        Line2D([0], [0], color="gray", linewidth=1.6, marker="o",
               linestyle="-", label=f"empirical ($n={n_draws}$)"),
        Line2D([0], [0], color="gray", linewidth=1.2, linestyle="--",
               label="analytical"),
    ]
    ax.legend(method_h + style_h, method_l + [h.get_label() for h in style_h],
              loc="upper left", fontsize=7, frameon=True, ncol=1)
    fig.tight_layout()

    pdf_path = os.path.join(out_dir, f"sensitivity_{topology}_{statistic}.pdf")
    png_path = os.path.join(out_dir, f"sensitivity_{topology}_{statistic}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Plot  -> {pdf_path}")
    print(f"  Plot  -> {png_path}")


# ── output: delay-experiment LaTeX tables ───────────────────────────────────

def _fmt_delay(x):
    """Compact scientific-notation formatter for delay magnitudes."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if x == 0:
        return "0"
    if abs(x) >= 1e5 or abs(x) < 1e-2:
        return f"{x:.2e}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.2f}"


def emit_latex_delays(df, out_dir, topology, pr, n_draws, ptype="lognormal"):
    """Emit two LaTeX tables, one per delay metric (pipelined, serial).
    Each table has rows = three methods and columns = sigma levels;
    cells are the mean over draws of the per-cell delay.
    """
    spec = PERTURBATIONS[ptype]
    levels = spec["levels"]
    pname  = spec["param_name"]

    methods = [
        ("CaMP-frozen-TCC", r"\Problemname-frozen ($\penbr$)"),
        ("CaMP-frozen-AOP", r"\Problemname-frozen ($\pencr$)"),
        ("CaP-frozen",      r"\Baseline-frozen"),
    ]
    metric_specs = [
        ("pipelined",
         r"Pipelined latency $\max_l d_l$",
         "pipelined latency"),
        ("serial",
         r"Serial latency $\sum_l d_l$",
         "serial latency"),
    ]

    for metric, metric_header, metric_caption in metric_specs:
        body_lines = []
        for method_key, method_label in methods:
            sub = df[(df.perturbation == ptype) &
                     (df.method == method_key) &
                     (df.metric == metric)]
            means = sub.groupby("param")["value"].mean()
            cells = " & ".join(_fmt_delay(means.get(lv, np.nan))
                               for lv in levels)
            body_lines.append(f"{method_label} & {cells}\\\\")
        body = "\n".join(body_lines)

        n_cols = len(levels)
        col_spec = "l|" + "c" * n_cols
        header_cells = " & ".join(f"${pname}={lv}$" for lv in levels)

        tex = (
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\resizebox{\\columnwidth}{!}{%\n"
            f"\\begin{{tabular}}{{{col_spec}}}\n"
            "\\toprule\n"
            f"\\textbf{{Method}} & \\multicolumn{{{n_cols}}}{{c}}"
            f"{{\\textbf{{{metric_header} (mean over {n_draws} draws)}}}}\\\\\n"
            f"\\cline{{2-{n_cols + 1}}}\n"
            f"& {header_cells}\\\\\n"
            "\\midrule\n"
            f"{body}\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "}\n"
            "\\caption{Mean " + metric_caption + " for the \\emph{"
            + topology + "} topology at $\\text{pr}=" + str(pr) + "$. "
            "The per-link cost $c_{m,m'}=1/\\text{bandwidth}_{m,m'}$ is "
            "log-normal with mean $\\C_{m,m'}$ and relative std $\\sigma$; "
            "the per-layer delay is the max over machine pairs of "
            "$\\text{bytes}_l[m,m'] \\cdot c_{m,m'}$. Delay units are bytes "
            "$\\times$ the units of $\\C$. Values averaged over "
            + str(n_draws) + " perturbation draws.}\n"
            f"\\label{{tab:delay_{topology}_{metric}}}\n"
            "\\end{table}\n"
        )

        path = os.path.join(out_dir, f"delay_{topology}_{metric}.tex")
        with open(path, "w") as f:
            f.write(tex)
        print(f"  LaTeX -> {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",
                   choices=["eval", "train_oracles", "all", "delays"],
                   default="eval",
                   help="'eval'/'train_oracles'/'all' run the realized-cost "
                        "sensitivity analysis. 'delays' runs the inference-"
                        "delay experiment: c=1/bandwidth on each link is "
                        "log-normal with mean C, per-layer delay is the max "
                        "over machine pairs of bytes*c, and pipelined "
                        "(max_l) and serial (sum_l) latencies are reported.")

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

    p.add_argument("--perturbations", default="lognormal",
                   help="Comma-separated subset of perturbation families to "
                        "run. Only 'lognormal' (debiased zero-mean) is "
                        "registered; the speed-class-reshuffle and "
                        "single-link-spike families were dropped per the "
                        "agreed appendix scope.")
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

    # Normalize --perturbations into a validated list, preserving CLI order.
    requested = [s.strip() for s in args.perturbations.split(",") if s.strip()]
    unknown = [name for name in requested if name not in PERTURBATIONS]
    if unknown:
        raise SystemExit(
            f"Unknown perturbation(s): {unknown}. "
            f"Valid choices: {list(PERTURBATIONS)}")
    if not requested:
        raise SystemExit("--perturbations cannot be empty.")
    args.perturbations = requested

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
        df, active_weights, base_maps = run_eval(args)
        emit_csv(df, args.out_dir, args.topology)
        # Denser sigma grid for the smooth analytical curves; superset of
        # the empirical SIGMA_LEVELS so the dashed line passes through the
        # marker positions exactly.
        sigmas_dense = sorted(
            set(float(s) for s in SIGMA_LEVELS).union(
                {0.0} | set(np.geomspace(0.005, 10.0, 60))))
        for ptype in args.perturbations:
            analyticals_std, analyticals_mad = {}, {}
            an_rng = np.random.default_rng(args.seed + 9_999)
            for method, w in active_weights.items():
                std_d, mad_d = analytical_relative_moments(
                    w, base_maps, sigmas_dense, n_mc=20000, rng=an_rng)
                analyticals_std[method] = std_d
                analyticals_mad[method] = mad_d
            emit_plot(df, args.out_dir, args.topology, ptype,
                      args.pr, args.n_draws, "std", analyticals_std)
            emit_plot(df, args.out_dir, args.topology, ptype,
                      args.pr, args.n_draws, "mad", analyticals_mad)
            emit_plot(df, args.out_dir, args.topology, ptype,
                      args.pr, args.n_draws, "mean", {})

    if args.mode == "delays":
        missing = [name for name, val in
                   (("--camp_aop_dir", args.camp_aop_dir),
                    ("--cap_dir",      args.cap_dir)) if val is None]
        if missing:
            raise SystemExit(f"{', '.join(missing)} required for --mode delays.")
        df_delays = run_delays(args)
        # Reuse emit_csv for the long-form record; the delay file has the
        # extra 'metric' column.
        delays_csv = os.path.join(args.out_dir,
                                  f"delays_{args.topology}.csv")
        df_delays.to_csv(delays_csv, index=False)
        print(f"  CSV   -> {delays_csv}")
        for ptype in args.perturbations:
            emit_latex_delays(df_delays, args.out_dir, args.topology,
                              args.pr, args.n_draws, ptype=ptype)


if __name__ == "__main__":
    main()
