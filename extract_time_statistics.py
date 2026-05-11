
"""
CaP2 Timing Data Extractor
===========================

For Combinations: 
    1. aggregate_partition_rows + partition_row
    2. full + partition_row

Reads a CaP2 SLURM log file and extracts timing statistics into:
  - <prefix>.txt          : human-readable summary (mirrors abilene_cost_1.txt style)
  - <prefix>_training_times.csv
  - <prefix>_retrain_times.csv
  - <prefix>_update_assignment_times.csv  (always empty for this log format)
  - <prefix>_elapsed_time_epoch0.csv
  - <prefix>_elapsed_time_epoch<N>.csv    (N = admm_epochs)

  Stores the above files into time_statistics folder

Usage:
    python extract_time_statistics.py <logfile> <prefix>
    

Arguments:
    logfile         Path to the SLURM .out log file
    prefix        Output file prefix
"""

import re
import csv
import sys
import os
from collections import defaultdict

def extract_log_info(filepath):
    training_time_pat = re.compile(r'@@@: Training time per epoch is ([\d.]+)s\.')
    update_time_pat   = re.compile(r'@@@: update_assignment time is ([\d.]+)s\.')
    epoch_pat         = re.compile(r'Epoch-\[(\d+)\]: Test loss:')
    partition_pat     = re.compile(r'Partition saved to .+?/(.+?)/partition_epoch_(\d+)')
    lr_pat            = re.compile(r'Learning rate:')
    elapsed_pat       = re.compile(r'Elapsed Time:\s*([\d.]+)s')

    training_times  = defaultdict(list)   # partition -> [(epoch, time)]
    update_times    = defaultdict(list)   # partition -> [(epoch, time)]
    retrain_times   = defaultdict(list)   # partition -> [(epoch, time)]
    elapsed_ep100   = {}                  # partition -> elapsed_time (epoch 100 only)
    elapsed_ep0     = {}                  # partition -> elapsed_time (epoch 0 only)

    pending_training_time = None
    pending_update_time   = None
    pending_epoch         = None
    current_partition     = None
    last_saved_epoch      = None          # epoch from the last "Partition saved" line
    logging_partition     = None
    logging_epoch         = None

    with open(filepath, 'r', errors='replace', newline='') as f:
        for line in f:
            line = line.strip().rstrip('\r')

            # Elapsed time (inside LOGGING block) — only record for epoch 100
            m = elapsed_pat.search(line)
            if m:
                if logging_epoch == 0 and logging_partition:
                    elapsed_ep0[logging_partition] = float(m.group(1))
                elif logging_epoch == 100 and logging_partition:
                    elapsed_ep100[logging_partition] = float(m.group(1))
                continue

            m = update_time_pat.search(line)
            if m:
                pending_update_time = float(m.group(1))
                continue

            m = training_time_pat.search(line)
            if m:
                pending_training_time = float(m.group(1))
                continue

            m = epoch_pat.search(line)
            if m:
                pending_epoch = int(re.search(r'\[(\d+)\]', line).group(1))
                continue

            # LOGGING separator — epoch & partition come from the just-processed Partition saved line
            if '+++++++++++++ LOGGING' in line:
                logging_partition = current_partition
                logging_epoch     = last_saved_epoch
                continue

            # Normal training block: ends with "Partition saved to .../partition_epoch_N"
            m = partition_pat.search(line)
            if m:
                folder    = m.group(1)
                epoch     = int(m.group(2))
                pr_match  = re.search(r'(pr[\d.]+_.+)', folder)
                partition = pr_match.group(1) if pr_match else folder
                partition = re.sub(r'_\d+$', '', partition)
                current_partition = partition
                last_saved_epoch  = epoch          # remember for the upcoming LOGGING block

                if pending_training_time is not None:
                    training_times[partition].append((epoch, pending_training_time))
                    pending_training_time = None

                if pending_update_time is not None:
                    update_times[partition].append((epoch, pending_update_time))
                    pending_update_time = None

                pending_epoch = None
                continue

            # Retrain block: ends with "Learning rate:"
            if lr_pat.search(line):
                if pending_training_time is not None and pending_epoch is not None and current_partition is not None:
                    retrain_times[current_partition].append((pending_epoch, pending_training_time))
                pending_training_time = None
                pending_epoch         = None
                continue

    return training_times, update_times, retrain_times, elapsed_ep0, elapsed_ep100


def write_csv(data, filename, time_label):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['partition', 'epoch', time_label])
        for partition in sorted(data):
            for epoch, t in sorted(data[partition]):
                writer.writerow([partition, epoch, t])
    print(f"Saved {filename}")


def write_elapsed_csv(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['partition', 'elapsed_time_epoch100_s'])
        for partition in sorted(data):
            writer.writerow([partition, data[partition]])
    print(f"Saved {filename}")


def print_summary(label, data, fh=None):
    def out(s): 
        print(s)
        if fh: fh.write(s + "\n")
    out(f"\n=== {label} ===")
    if not data:
        out("  (none found)")
        return
    for p in sorted(data):
        if isinstance(data[p], list):
            entries = data[p]
            total = sum(t for _, t in entries)
            avg   = total / len(entries) if entries else 0
            out(f"  {p}")
            out(f"    {len(entries)} entries, avg {avg:.2f}s, total {total:.2f}s")
        else:
            out(f"  {p}  ->  {data[p]:.4f}s")

# Extracts timing statistics from SLURM/experiment log files produced by distributed
# training runs. Parses multiple partitions from a single log file and outputs one
# CSV per metric plus a human-readable summary.

# To Run: 
# python extract_time_statistics.py slurm-5426250.out <prefix>
# <prefix>: Label prepended to all output filenames e.g. babarasi_cost

# saves under time_statistics folder:
# === Training times per epoch : e.g 100 training epochs (admm_epochs)
# === Update assignment times : e.g assigment times that take place every after 5 training epochs (theta_epochs)
# === Retrain times per epoch : e.g 90 retraining epochs (retrain_ep)
# === Elapsed time at epoch 0 : Elapsed time, take only those with pruning ratio: 0.0
# === Elapsed time at epoch 100 === (Cummulative) Elapsed time at the end of the training 

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python extract_log.py <logfile> <prefix>")
        print("  e.g: python extract_log.py slurm-5426250.out exp1")
        sys.exit(1)

    logfile = sys.argv[1]
    prefix  = sys.argv[2]

    print(f"Reading: {logfile}")
    print(f"Output prefix: {prefix}")
    training_times, update_times, retrain_times, elapsed_ep0, elapsed_ep100 = extract_log_info(logfile)

    all_partitions = sorted(set(
        list(training_times.keys()) +
        list(update_times.keys()) +
        list(retrain_times.keys()) +
        list(elapsed_ep0.keys()) +
        list(elapsed_ep100.keys())
    ))
    print(f"\nFound {len(all_partitions)} unique partition(s).")

    out_dir = 'time_statistics'+"/"+prefix
    os.makedirs(out_dir, exist_ok=True)
    summary_file = os.path.join(out_dir, f'{prefix}.txt')
    with open(summary_file, 'w', encoding='utf-8') as fh:
        header = f"Summary for: {logfile}  |  prefix: {prefix}"
        print(header); fh.write(header + "\n")
        print_summary("Training times per epoch", training_times, fh)
        print_summary("Update assignment times",  update_times,   fh)
        print_summary("Retrain times per epoch",  retrain_times,  fh)
        print_summary("Elapsed time at epoch 0",   elapsed_ep0,   fh)
        print_summary("Elapsed time at epoch 100", elapsed_ep100, fh)
    print(f"\nSaved {summary_file}")

    write_csv(training_times, os.path.join(out_dir, f'{prefix}_training_times.csv'),          'training_time_s')
    write_csv(update_times,   os.path.join(out_dir, f'{prefix}_update_assignment_times.csv'),  'update_assignment_time_s')
    write_csv(retrain_times,  os.path.join(out_dir, f'{prefix}_retrain_times.csv'),           'retrain_time_s')
    write_elapsed_csv(elapsed_ep0,   os.path.join(out_dir, f'{prefix}_elapsed_time_epoch0.csv'))
    write_elapsed_csv(elapsed_ep100, os.path.join(out_dir, f'{prefix}_elapsed_time_epoch100.csv'))
