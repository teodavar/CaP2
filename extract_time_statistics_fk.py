#!/usr/bin/env python3
"""
CaP2 Timing Data Extractor
===========================

For Combination: 
    1. full + kernel


Reads a CaP2 SLURM log file and extracts timing statistics into:
  - <prefix>.txt          : human-readable summary (mirrors abilene_cost_1.txt style)
  - <prefix>_training_times.csv
  - <prefix>_retrain_times.csv
  - <prefix>_update_assignment_times.csv  (always empty for this log format)
  - <prefix>_elapsed_time_epoch0.csv
  - <prefix>_elapsed_time_epoch<N>.csv    (N = admm_epochs)

USE:
python extract_time_statistics_fk.py ./barabasi/slurm-5447253_2.out --prefix barabasi_3

Usage:

Arguments:
    logfile         Path to the SLURM .out log file
    --prefix        Output file prefix (default: derived from log filename)

How the parser works
--------------------
Each "run" (one pruning ratio) is delimited by:
  - Start: a dict line like {'conv1.weight': 0.X, ...}  followed by
           "!!!! ADMM runs with: original full kernel"
    OR an explicit "=== Progressive Step N / M  |  Prune ratio: X.X ===" line.
  - The experiment name is parsed from the "Partition saved to ..." line.
  - ADMM training epochs: lines containing "@@@: Training time per epoch is XX.XXs."
    that appear BEFORE "Progress saved. Exiting gracefully." (marks end of ADMM).
  - Retrain training epochs: same "@@@:" lines that appear AFTER the retrain boundary
    which is identified by "Epoch-[000]: Test loss" + "Save model" / "Learning rate:"
    pattern (no LOGGING block).
  - Elapsed time at epoch 0: parsed from the first "📌 **Epoch 0 Summary**" block
    (🕒 Elapsed Time: X.Xs).
  - Elapsed time at epoch N: parsed from the "📌 **Epoch N Summary**" block where
    N == admm_epochs (last ADMM epoch).
  - Update assignment times: EMPTY for this log format (section preserved for
    compatibility).
"""

import re
import os
import csv
import sys
import argparse
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
RE_PROGRESSIVE_STEP = re.compile(
    r"=== Progressive Step\s+(\d+)\s*/\s*(\d+)\s*\|\s*Prune ratio:\s*([\d.]+)\s*==="
)
RE_PARTITION_SAVED = re.compile(
    r"Partition saved to (.+/partition_final)"
)
RE_TRAINING_TIME = re.compile(
    r"@@@: Training time per epoch is ([\d.]+)s\."
)
RE_EPOCH_SUMMARY_START = re.compile(
    r"📌 \*\*Epoch (\d+) Summary\*\*"
)
RE_ELAPSED_TIME = re.compile(
    r"🕒 Elapsed Time:\s*([\d.]+)s"
)
RE_ADMM_END = re.compile(
    r"Progress saved\. Exiting gracefully\."
)
# Retrain boundary: "Epoch-[000]: Test loss" followed (within a few lines) by
# "Save model" or "Learning rate:" WITHOUT a LOGGING block in between.
RE_EPOCH_LINE = re.compile(
    r"Epoch-\[(\d+)\]: Test loss"
)
RE_LOGGING = re.compile(r"\+{10,}\s*LOGGING\s*\+{10,}")
RE_SAVE_MODEL = re.compile(r"^Save model\s*$")
RE_LEARNING_RATE = re.compile(r"^Learning rate:")


def extract_run_name_from_partition(partition_path: str) -> str:
    """
    Convert a partition_final path like:
      ./experiment_logs_barabasi_albert/cifar100_resnet101_pr0.5_np10_kernel_BarabasiAlbert_fixed_original_full_60/partition_final
    into a compact run key like:
      pr0.5_np10_kernel_BarabasiAlbert_fixed_original_full
    """
    parts = partition_path.rstrip("/").split("/")
    # The folder is the second-to-last component
    folder = parts[-2]
    # folder example: cifar100_resnet101_pr0.5_np10_kernel_BarabasiAlbert_fixed_original_full_60
    # Strip leading model/dataset prefix up to first "pr" token
    match = re.search(r"(pr[\d.]+_.+?)(?:_\d+)?$", folder)
    if match:
        return match.group(1)
    return folder


def parse_log(filepath: str):
    """
    Main parser.  Returns a dict with keys:
        log_filename  : str
        admm_epochs   : int   (from config, or inferred)
        runs          : list of run dicts, each containing:
            name              : str  (e.g. pr0.5_np10_...)
            prune_ratio       : float
            training_times    : list of float   (ADMM epochs)
            retrain_times     : list of float
            elapsed_epoch0    : float or None
            elapsed_epochN    : float or None   (N = admm_epochs)
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    # ---- First pass: read config (admm_epochs) ----
    admm_epochs = None
    for line in lines[:200]:
        m = re.match(r"admm_epochs\s*:\s*(\d+)", line.strip())
        if m:
            admm_epochs = int(m.group(1))
            break
    if admm_epochs is None:
        admm_epochs = 60  # fallback default

    # ---- Second pass: segment the file into runs ----
    # A "run" starts when we see the prune ratio dict / ADMM header.
    # We'll use "Partition saved to ..." as the run name anchor.
    # Strategy: walk line by line, tracking state.

    runs = []
    current_run = None
    in_admm = False         # True while collecting ADMM training times
    in_retrain = False      # True while collecting retrain times
    retrain_boundary_candidate = None  # epoch number seen just before Save model
    admm_ended = False      # flipped by "Progress saved. Exiting gracefully."
    pending_epoch_num = None
    epoch_summary_epoch = None

    def new_run(prune_ratio=None):
        return {
            "name": None,
            "prune_ratio": prune_ratio,
            "training_times": [],
            "retrain_times": [],
            "elapsed_epoch0": None,
            "elapsed_epochN": None,
        }

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip("\r\n")

        # --- Detect start of a new progressive step (explicit marker) ---
        m = RE_PROGRESSIVE_STEP.match(line.strip())
        if m:
            prune_ratio = float(m.group(3))
            if current_run is not None:
                runs.append(current_run)
            current_run = new_run(prune_ratio)
            in_admm = True
            in_retrain = False
            admm_ended = False
            i += 1
            continue

        # --- Detect implicit start of a new ADMM step (pr0.0 has no explicit marker
        #     but begins with a dict line + "!!!! ADMM runs with")
        if "!!!! ADMM runs with:" in line and current_run is None:
            current_run = new_run()
            in_admm = True
            in_retrain = False
            admm_ended = False
            i += 1
            continue

        # Second case: step 2 (pr=0.5) starts right after step 1 partition with
        # a new dict line — we catch this via the dict having a different pr value
        # BUT since we won't re-parse the dict, we rely on the Epoch-[000] + no LOGGING
        # heuristic below for retrain detection.

        # --- Partition saved → extract run name ---
        m = RE_PARTITION_SAVED.search(line)
        if m and current_run is not None:
            current_run["name"] = extract_run_name_from_partition(m.group(1))
            i += 1
            continue

        # --- ADMM end marker ---
        if RE_ADMM_END.search(line):
            admm_ended = True
            in_admm = False
            in_retrain = False
            i += 1
            continue

        # --- Training time line (@@@) ---
        m = RE_TRAINING_TIME.search(line)
        if m and current_run is not None:
            t = float(m.group(1))
            if in_admm and not admm_ended:
                current_run["training_times"].append(t)
            elif in_retrain or admm_ended:
                current_run["retrain_times"].append(t)
                in_retrain = True
            i += 1
            continue

        # --- Detect retrain boundary:
        #     "Epoch-[000]: Test loss" that is NOT followed by "LOGGING" but IS
        #     followed by "Save model" or "Learning rate:" within next ~3 lines.
        m = RE_EPOCH_LINE.search(line)
        if m:
            epoch_num = int(m.group(1))
            if epoch_num == 0 and admm_ended and current_run is not None and not in_retrain:
                # Look ahead up to 5 lines
                has_logging = False
                has_retrain_marker = False
                for j in range(i + 1, min(i + 6, n)):
                    if RE_LOGGING.search(lines[j]):
                        has_logging = True
                        break
                    if RE_SAVE_MODEL.match(lines[j].strip()) or RE_LEARNING_RATE.match(lines[j].strip()):
                        has_retrain_marker = True
                        break
                if has_retrain_marker and not has_logging:
                    in_retrain = True
            i += 1
            continue

        # --- Epoch summary block ---
        m = RE_EPOCH_SUMMARY_START.search(line)
        if m:
            epoch_summary_epoch = int(m.group(1))
            i += 1
            continue

        # --- Elapsed time inside a summary block ---
        m = RE_ELAPSED_TIME.search(line)
        if m and current_run is not None and epoch_summary_epoch is not None:
            elapsed = float(m.group(1))
            if epoch_summary_epoch == 0:
                if current_run["elapsed_epoch0"] is None:
                    current_run["elapsed_epoch0"] = elapsed
            elif epoch_summary_epoch == admm_epochs:
                current_run["elapsed_epochN"] = elapsed
            epoch_summary_epoch = None
            i += 1
            continue

        i += 1

    # Don't forget the last run
    if current_run is not None:
        runs.append(current_run)

    # Fill in missing names with fallback
    for idx, run in enumerate(runs):
        if run["name"] is None:
            run["name"] = f"run_{idx}_pr{run['prune_ratio']}"

    return {
        "log_filename": filepath.name,
        "admm_epochs": admm_epochs,
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def fmt_stats(times: list) -> str:
    if not times:
        return "(no data)"
    n = len(times)
    avg = sum(times) / n
    total = sum(times)
    return f"{n} entries, avg {avg:.2f}s, total {total:.2f}s"


def write_txt(data: dict, outpath: str, prefix: str, log_source: str):
    admm_epochs = data["admm_epochs"]
    runs = data["runs"]

    lines = []
    lines.append(f"Summary for: {log_source}  |  prefix: {prefix}\n")
    lines.append("\n")

    # === Training times per epoch ===
    lines.append("=== Training times per epoch ===\n")
    for run in runs:
        t = run["training_times"]
        lines.append(f"  {run['name']}\n")
        lines.append(f"    {fmt_stats(t)}\n")
    lines.append("\n")

    # === Update assignment times === (always empty)
    lines.append("=== Update assignment times ===\n")
    lines.append("\n")

    # === Retrain times per epoch ===
    lines.append("=== Retrain times per epoch ===\n")
    for run in runs:
        t = run["retrain_times"]
        if t:
            lines.append(f"  {run['name']}\n")
            lines.append(f"    {fmt_stats(t)}\n")
    lines.append("\n")

    # === Elapsed time at epoch 0 ===
    lines.append("=== Elapsed time at epoch 0 ===\n")
    for run in runs:
        e = run["elapsed_epoch0"]
        if e is not None:
            lines.append(f"  {run['name']}  ->  {e:.4f}s\n")
    lines.append("\n")

    # === Elapsed time at epoch N ===
    lines.append(f"=== Elapsed time at epoch {admm_epochs} ===\n")
    for run in runs:
        e = run["elapsed_epochN"]
        if e is not None:
            lines.append(f"  {run['name']}  ->  {e:.4f}s\n")
    lines.append("\n")

    with open(outpath, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    print(f"  Written: {outpath}")


def write_csv_times(times_by_run: dict, outpath: str, column_name: str):
    """Write a CSV where each column is a run name, rows are epoch indices."""
    if not times_by_run:
        # Write empty file with header
        with open(outpath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([column_name])
        print(f"  Written (empty): {outpath}")
        return

    max_len = max(len(v) for v in times_by_run.values()) if times_by_run else 0
    headers = [column_name] + list(times_by_run.keys())
    rows = []
    for i in range(max_len):
        row = [i + 1]
        for run_name in times_by_run:
            vals = times_by_run[run_name]
            row.append(f"{vals[i]:.4f}" if i < len(vals) else "")
        rows.append(row)

    with open(outpath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  Written: {outpath}")


def write_csv_elapsed(elapsed_by_run: dict, outpath: str):
    """Write a CSV with run names and single elapsed time values."""
    with open(outpath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_name", "elapsed_time_s"])
        for run_name, elapsed in elapsed_by_run.items():
            writer.writerow([run_name, f"{elapsed:.4f}" if elapsed is not None else ""])
    print(f"  Written: {outpath}")


def write_outputs(data: dict, outdir: str, prefix: str, log_source: str):
    os.makedirs(outdir, exist_ok=True)
    runs = data["runs"]
    admm_epochs = data["admm_epochs"]

    # Build dicts for CSV writers
    training_times = {r["name"]: r["training_times"] for r in runs}
    retrain_times  = {r["name"]: r["retrain_times"]  for r in runs if r["retrain_times"]}
    elapsed_epoch0 = {r["name"]: r["elapsed_epoch0"] for r in runs if r["elapsed_epoch0"] is not None}
    elapsed_epochN = {r["name"]: r["elapsed_epochN"] for r in runs if r["elapsed_epochN"] is not None}

    # .txt summary
    txt_path = os.path.join(outdir, f"{prefix}.txt")
    write_txt(data, txt_path, prefix, log_source)

    # training times CSV
    write_csv_times(
        training_times,
        os.path.join(outdir, f"{prefix}_training_times.csv"),
        "epoch"
    )

    # retrain times CSV
    write_csv_times(
        retrain_times,
        os.path.join(outdir, f"{prefix}_retrain_times.csv"),
        "epoch"
    )

    # update assignment times CSV (always empty for this format)
    write_csv_times(
        {},
        os.path.join(outdir, f"{prefix}_update_assignment_times.csv"),
        "epoch"
    )

    # elapsed time at epoch 0 CSV
    write_csv_elapsed(
        elapsed_epoch0,
        os.path.join(outdir, f"{prefix}_elapsed_time_epoch0.csv")
    )

    # elapsed time at epoch N CSV
    write_csv_elapsed(
        elapsed_epochN,
        os.path.join(outdir, f"{prefix}_elapsed_time_epoch{admm_epochs}.csv")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract CaP2 timing data from a SLURM log file."
    )
    parser.add_argument("logfile", help="Path to the SLURM .out log file")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output file prefix (default: derived from log filename)"
    )

    args = parser.parse_args()

    logfile = Path(args.logfile)
    if not logfile.exists():
        print(f"Error: file not found: {logfile}", file=sys.stderr)
        sys.exit(1)

    prefix = args.prefix or logfile.stem
    outdir = 'time_statistics'+"/"+prefix

    print(f"Parsing: {logfile}")
    data = parse_log(str(logfile))

    print(f"Found {len(data['runs'])} run(s), admm_epochs={data['admm_epochs']}")
    for r in data["runs"]:
        print(f"  [{r['name']}]  train={len(r['training_times'])}  "
              f"retrain={len(r['retrain_times'])}  "
              f"e0={r['elapsed_epoch0']}  "
              f"eN={r['elapsed_epochN']}")

    print(f"\nWriting outputs to: {outdir}  (prefix={prefix})")
    write_outputs(data, outdir, prefix, str(logfile))
    print("\nDone.")


if __name__ == "__main__":
    main()
