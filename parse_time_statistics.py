"""
parse_time_statistics.py
Parses a CaP2 time-statistics .txt file and writes an Excel workbook with
two sheets:

  row_output  – flat table (one row per topology/penalty/sparsity combination)
  output1     – pivot-style table grouped by topology, with A/B/C sub-columns

Usage:
    python parse_time_statistics.py <input.txt> <output.xlsx>
"""

import re
import sys
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_key(name):
    """
    Extract (topology, penalty, sparsity, pr) from a run name.

    Normal:  pr0.75_np11_partition_row_Abilene_cost_rsgn5_original_aggregate_partition_rows
             → topology=Abilene_cost, penalty=aggregate_partition_rows, sparsity=partition_row

    Fixed:   pr0.0_np11_kernel_Abilene_cost_rsgn5_fixed_original_full
             → topology=Abilene_cost, penalty=full, sparsity=kernel
    """
    pr_m = re.match(r'pr([\d.]+)', name)
    pr = pr_m.group(1) if pr_m else None

    body = re.sub(r'^pr[\d.]+_np\d+_', '', name)
    fixed = bool(re.search(r'(?:_rsgn\d+)?_fixed_original_', body))
    rsgn_split = re.split(r'(?:_rsgn\d+)?_fixed_original_|_rsgn\d+_original_', body, maxsplit=1)
    meta   = rsgn_split[0]
    suffix = rsgn_split[1] if len(rsgn_split) > 1 else ''

    # For fixed entries: penalty=full, sparsity=kernel (regardless of meta prefix)
    if fixed:
        penalty  = 'full'
        sparsity = 'kernel'
        parts = meta.split('_')
        topo_start = next((i for i, p in enumerate(parts) if p and p[0].isupper()), None)
        topology = '_'.join(parts[topo_start:]) if topo_start is not None else meta
    else:
        penalty = suffix
        parts = meta.split('_')
        topo_start = next((i for i, p in enumerate(parts) if p and p[0].isupper()), None)
        if topo_start is not None:
            sparsity = '_'.join(parts[:topo_start])
            topology = '_'.join(parts[topo_start:])
        else:
            sparsity = meta
            topology = ''

    return {'pr': pr, 'topology': topology, 'penalty': penalty, 'sparsity': sparsity}


def parse_txt(path):
    data = defaultdict(lambda: defaultdict(dict))
    section_map = {
        'Training times per epoch':  'training',
        'Update assignment times':   'assignment',
        'Retrain times per epoch':   'retrain',
        'Elapsed time at epoch 0':   'elapsed_epoch0',
        'Elapsed time at epoch 100': 'elapsed_final',
    }
    current_section = None
    current_name    = None
    section_re = re.compile(r'===\s*(.+?)\s*===')
    name_re    = re.compile(r'^\s{2}(\S+)')
    stat_re    = re.compile(r'(\d+) entries,\s*avg\s+([\d.]+)s,\s*total\s+([\d.]+)s')
    elapsed_re = re.compile(r'^\s{2}(\S+)\s+->\s+([\d.]+)s')

    with open(path, encoding='utf-8') as fh:
        for raw in fh:
            line = raw.rstrip('\n')
            sm = section_re.search(line)
            if sm:
                label = sm.group(1).strip()
                current_section = None
                for key, val in section_map.items():
                    if key in label:
                        current_section = val
                        break
                current_name = None
                continue
            if current_section is None:
                continue
            if current_section in ('elapsed_epoch0', 'elapsed_final'):
                em = elapsed_re.match(line)
                if em:
                    name, val = em.group(1), float(em.group(2))
                    info = parse_key(name)
                    key  = (info['topology'], info['penalty'], info['sparsity'])
                    data[key][info['pr']][current_section] = val
                continue
            nm = name_re.match(line)
            if nm:
                current_name = nm.group(1)
                continue
            sm2 = stat_re.search(line)
            if sm2 and current_name:
                avg = float(sm2.group(2))
                ttl = float(sm2.group(3))
                info = parse_key(current_name)
                key  = (info['topology'], info['penalty'], info['sparsity'])
                data[key][info['pr']][current_section + '_avg'] = avg
                data[key][info['pr']][current_section + '_ttl'] = ttl
                current_name = None
    return data


# ── style helpers ─────────────────────────────────────────────────────────────

SPARSITY_ORDER = {'partition_row': 0, 'kernel': 1}

def row_sort_key(item):
    (topology, penalty, sparsity), _ = item
    return (topology, penalty, SPARSITY_ORDER.get(sparsity, 99))

def header_style(cell, bold=True):
    cell.font      = Font(bold=bold, size=10)
    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

def data_style(cell, bold=False, align='center'):
    cell.font      = Font(bold=bold, size=10)
    cell.alignment = Alignment(horizontal=align, vertical='center')


# ── TAB 1: row_output ─────────────────────────────────────────────────────────

def build_row_output(wb, data):
    ws = wb.create_sheet('row_output')

    header1 = ['Topology', 'Penalty', 'Sparsity', 'pr0.0',
               'pr0.5',  '', '', '', '', '', '',
               'pr0.75', '', '', '', '', '', '',
               'pr0.85', '', '', '', '', '', '',
               'pr1.0',  '', '', '', '', '', '']
    header2 = ['', '', '', '',
               'Training', '', 'Elapsed Training', 'Assignment', '', 'Retraining', '',
               'Training', '', 'Elapsed Training', 'Assignment', '', 'Retraining', '',
               'Training', '', 'Elapsed Training', 'Assignment', '', 'Retraining', '',
               'Training', '', 'Elapsed Training', 'Assignment', '', 'Retraining', '']
    header3 = ['', '', '', '',
               'Avg', 'Ttl', 'Value', 'Avg', 'Ttl', 'Avg', 'Ttl',
               'Avg', 'Ttl', 'Value', 'Avg', 'Ttl', 'Avg', 'Ttl',
               'Avg', 'Ttl', 'Value', 'Avg', 'Ttl', 'Avg', 'Ttl',
               'Avg', 'Ttl', 'Value', 'Avg', 'Ttl', 'Avg', 'Ttl']

    PR_COLS = [5, 12, 19, 26]   # 1-indexed start cols for pr0.5/0.75/0.85/1.0

    for h in [header1, header2, header3]:
        ws.append(h)

    # Style all header cells (bold, centered — no colour)
    for r in range(1, 4):
        for col_idx in range(1, len(header1) + 1):
            header_style(ws.cell(r, col_idx))

    # Merges
    ws.merge_cells('A1:A3'); ws.merge_cells('B1:B3')
    ws.merge_cells('C1:C3'); ws.merge_cells('D1:D3')
    for sc in PR_COLS:
        ws.merge_cells(f'{get_column_letter(sc)}1:{get_column_letter(sc+6)}1')
        ws.merge_cells(f'{get_column_letter(sc)}2:{get_column_letter(sc+1)}2')
        ws.merge_cells(f'{get_column_letter(sc+3)}2:{get_column_letter(sc+4)}2')
        ws.merge_cells(f'{get_column_letter(sc+5)}2:{get_column_letter(sc+6)}2')

    def pr_vals(d, pr):
        return [
            d.get(pr, {}).get('training_avg',   ''),
            d.get(pr, {}).get('training_ttl',   ''),
            d.get(pr, {}).get('elapsed_final',  ''),
            d.get(pr, {}).get('assignment_avg', ''),
            d.get(pr, {}).get('assignment_ttl', ''),
            d.get(pr, {}).get('retrain_avg',    ''),
            d.get(pr, {}).get('retrain_ttl',    ''),
        ]

    for row_i, ((topology, penalty, sparsity), pr_data) in enumerate(sorted(data.items(), key=row_sort_key)):
        elapsed0 = pr_data.get('0.0', {}).get('elapsed_epoch0', '')
        row = [topology, penalty, sparsity, elapsed0]
        for pr in ('0.5', '0.75', '0.85', '1.0'):
            row.extend(pr_vals(pr_data, pr))
        ws.append(row)
        excel_row = row_i + 4
        for col_idx in range(1, len(row) + 1):
            data_style(ws.cell(excel_row, col_idx))

    ws.column_dimensions['A'].width = 18
    ws.column_dimensions['B'].width = 26
    ws.column_dimensions['C'].width = 15
    ws.column_dimensions['D'].width = 10
    for ci in range(5, 33):
        ws.column_dimensions[get_column_letter(ci)].width = 11
    ws.freeze_panes = 'A4'


# ── TAB 2: output1 ────────────────────────────────────────────────────────────
#
# Columns:
#   1=Topology, 2=Total Time (ms)
#   3-5  pr=0.5  A/B/C
#   6-8  pr=0.75 A/B/C
#   9-11 pr=0.85 A/B/C
#   12-14 pr=1.0 A/B/C
#   15   pr=0.0
#
# A = aggregate_partition_rows / partition_row
# B = full / partition_row
# C = full / kernel
#
# Metric rows per topology: Training (Ttl), Assignment (Ttl), Retraining (Ttl), Elapsed Time
# pr0.0 column only shows a value on the Elapsed Time row (elapsed_epoch0)

def build_output1(wb, data):
    ws = wb.create_sheet('output1')

    PR_KEYS    = ['0.5', '0.75', '0.85', '1.0']
    PR_LBLS    = ['pr=0.5', 'pr=0.75', 'pr=0.85', 'pr=1.0']
    PR_START   = {'0.5': 3, '0.75': 6, '0.85': 9, '1.0': 12}
    COL_PR00   = 15          # pr=0.0 starts at col 15, spans A/B/C → cols 15/16/17
    LAST_COL   = 17
    VARIANTS = [
        ('aggregate_partition_rows', 'partition_row'),  # A
        ('full',                     'partition_row'),  # B
        ('full',                     'kernel'),          # C
    ]
    METRICS = ['Training', 'Assignment', 'Retraining', 'Elapsed Time']

    # ── Row 1: group headers ──
    ws.cell(1, 1).value = 'Topology'
    ws.cell(1, 2).value = 'Total Time (ms)'
    header_style(ws.cell(1, 1))
    header_style(ws.cell(1, 2))
    ws.merge_cells('A1:A2')
    ws.merge_cells('B1:B2')

    for pr_key, pr_lbl in zip(PR_KEYS, PR_LBLS):
        sc = PR_START[pr_key]
        c = ws.cell(1, sc)
        c.value = pr_lbl
        header_style(c)
        ws.merge_cells(start_row=1, start_column=sc, end_row=1, end_column=sc + 2)

    # pr=0.0 header spans cols 15-17
    c00 = ws.cell(1, COL_PR00)
    c00.value = 'pr=0.0'
    header_style(c00)
    ws.merge_cells(start_row=1, start_column=COL_PR00, end_row=1, end_column=LAST_COL)

    # ── Row 2: A/B/C sub-headers for all pr blocks including pr=0.0 ──
    for pr_key in PR_KEYS:
        sc = PR_START[pr_key]
        for i, letter in enumerate(['A', 'B', 'C']):
            c = ws.cell(2, sc + i)
            c.value = letter
            header_style(c)

    for i, letter in enumerate(['A', 'B', 'C']):
        c = ws.cell(2, COL_PR00 + i)
        c.value = letter
        header_style(c)

    # ── Data rows ──
    all_topologies = sorted(set(t for (t, p, s) in data.keys()))
    current_row = 3

    for topology in all_topologies:
        topo_start = current_row

        for metric in METRICS:
            bc = ws.cell(current_row, 2)
            bc.value = metric
            data_style(bc, bold=(metric == 'Training'), align='left')

            for pr_key in PR_KEYS:
                sc = PR_START[pr_key]
                for i, (penalty, sparsity) in enumerate(VARIANTS):
                    key  = (topology, penalty, sparsity)
                    pr_d = data.get(key, {}).get(pr_key, {})
                    if metric == 'Training':
                        val = pr_d.get('training_ttl', '')
                    elif metric == 'Assignment':
                        val = pr_d.get('assignment_ttl', '')
                    elif metric == 'Retraining':
                        val = pr_d.get('retrain_ttl', '')
                    else:
                        val = pr_d.get('elapsed_final', '')
                    c = ws.cell(current_row, sc + i)
                    c.value = val
                    data_style(c)

            # pr=0.0: A/B/C columns — only Elapsed Time row carries values
            for i, (penalty, sparsity) in enumerate(VARIANTS):
                c0 = ws.cell(current_row, COL_PR00 + i)
                if metric == 'Elapsed Time':
                    key = (topology, penalty, sparsity)
                    c0.value = data.get(key, {}).get('0.0', {}).get('elapsed_epoch0', '')
                data_style(c0)

            current_row += 1

        # Merge topology cell across its metric rows
        ws.merge_cells(start_row=topo_start, start_column=1,
                       end_row=current_row - 1, end_column=1)
        tc = ws.cell(topo_start, 1)
        tc.value = topology
        data_style(tc, bold=True)

    # ── Legend ──
    legend_row = current_row + 2
    lines = [
        'Legend:',
        'A: Penalty: aggregate_partition_rows',
        '    Sparsity: partition_row',
        'B: Penalty: full',
        '    Sparsity: partition_row',
        'C: Penalty: full',
        '    Sparsity: kernel',
    ]
    for offset, text in enumerate(lines):
        c = ws.cell(legend_row + offset, 1)
        c.value = text
        c.font  = Font(size=9, bold=(offset == 0))
        c.alignment = Alignment(horizontal='left', vertical='center')
        ws.merge_cells(start_row=legend_row + offset, start_column=1,
                       end_row=legend_row + offset, end_column=4)

    # ── Column widths ──
    ws.column_dimensions['A'].width = 18
    ws.column_dimensions['B'].width = 18
    for ci in range(3, LAST_COL + 1):
        ws.column_dimensions[get_column_letter(ci)].width = 13
    ws.freeze_panes = 'A3'



# ── LaTeX table (mirrors output1 layout) ─────────────────────────────────────
#
# Structure:
#   Col 1  : Topology   (multirow across 4 metric rows)
#   Col 2  : Metric     (Training / Assignment / Retraining / Elapsed Time)
#   Cols 3-5  : pr=0.5   A / B / C
#   Cols 6-8  : pr=0.75  A / B / C
#   Cols 9-11 : pr=0.85  A / B / C
#   Cols 12-14: pr=1.0   A / B / C
#   Cols 15-17: pr=0.0   A / B / C  (only Elapsed Time row has values)

def build_latex(data, tex_path):
    PR_KEYS  = ['0.5', '0.75', '0.85', '1.0', '0.0']
    PR_LBLS  = [r'$p_r=0.5$', r'$p_r=0.75$', r'$p_r=0.85$', r'$p_r=1.0$', r'$p_r=0.0$']
    VARIANTS = [
        ('aggregate_partition_rows', 'partition_row'),  # A
        ('full',                     'partition_row'),  # B
        ('full',                     'kernel'),          # C
    ]
    METRICS  = ['Training', 'Assignment', 'Retraining', 'Elapsed Time']

    def fmt(v):
        """Format a numeric value for LaTeX; blank if missing."""
        if v == '' or v is None:
            return '---'
        try:
            return f'{float(v):.2f}'
        except (ValueError, TypeError):
            return str(v)

    def get_val(data, topology, pr_key, penalty, sparsity, metric):
        key  = (topology, penalty, sparsity)
        pr_d = data.get(key, {}).get(pr_key, {})
        if pr_key == '0.0':
            return pr_d.get('elapsed_epoch0', '') if metric == 'Elapsed Time' else ''
        if metric == 'Training':
            return pr_d.get('training_ttl', '')
        elif metric == 'Assignment':
            return pr_d.get('assignment_ttl', '')
        elif metric == 'Retraining':
            return pr_d.get('retrain_ttl', '')
        else:  # Elapsed Time
            return pr_d.get('elapsed_final', '')

    all_topologies = sorted(set(t for (t, p, s) in data.keys()))

    # Total data columns: 1 (Topology) + 1 (Metric) + 5 pr-blocks × 3 variants = 17
    num_data_cols = 2 + len(PR_KEYS) * 3
    col_spec = 'll' + ('ccc' * len(PR_KEYS))  # l=left for topo+metric, c=center for values

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{CaP2 Time Statistics}')
    lines.append(r'  \label{tab:cap2_time_stats}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(f'  \begin{{tabular}}{{{col_spec}}}')
    lines.append(r'  \toprule')

    # ── Header row 1: pr-block labels ──
    pr_headers = ' & '.join(
        r'\multicolumn{3}{c}{' + lbl + '}' for lbl in PR_LBLS
    )
    lines.append(
        r'  \multirow{2}{*}{\textbf{Topology}} & '
        r'\multirow{2}{*}{\textbf{Metric}} & '
        + pr_headers + r' \\'
    )

    # cmidrule under each pr block (cols 3-5, 6-8, … 15-17)
    cmidrules = ' '.join(
        r'\cmidrule(lr){' + f'{3 + i*3}-{5 + i*3}' + '}'
        for i in range(len(PR_KEYS))
    )
    lines.append(f'  {cmidrules}')

    # ── Header row 2: A / B / C repeated ──
    abc_headers = ' & '.join(['A & B & C'] * len(PR_KEYS))
    lines.append(f'  & & {abc_headers} \\\\')
    lines.append(r'  \midrule')

    # ── Data rows ──
    for topo_idx, topology in enumerate(all_topologies):
        topo_label = topology.replace('_', r'\_')

        for m_idx, metric in enumerate(METRICS):
            # Topology cell: multirow spanning 4 metric rows, only on first metric
            if m_idx == 0:
                topo_cell = r'\multirow{4}{*}{\textbf{' + topo_label + '}}'
            else:
                topo_cell = ''

            # Value cells for each pr block
            value_cells = []
            for pr_key in PR_KEYS:
                for penalty, sparsity in VARIANTS:
                    v = get_val(data, topology, pr_key, penalty, sparsity, metric)
                    value_cells.append(fmt(v))

            row = f'  {topo_cell} & {metric} & ' + ' & '.join(value_cells) + r' \\'
            lines.append(row)

        # Separator between topologies (but not after the last one)
        if topo_idx < len(all_topologies) - 1:
            lines.append(r'  \midrule')

    lines.append(r'  \bottomrule')
    lines.append(r'  \end{tabular}%')
    lines.append(r'  }')  # end resizebox
    lines.append(r'\end{table}')
    lines.append('')

    # Legend as a LaTeX comment block
    lines.append(r'% Legend:')
    lines.append(r'% A: Penalty=aggregate\_partition\_rows, Sparsity=partition\_row')
    lines.append(r'% B: Penalty=full, Sparsity=partition\_row')
    lines.append(r'% C: Penalty=full, Sparsity=kernel')

    with open(tex_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f"LaTeX table written to {tex_path}")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python parse_time_statistics.py <input.txt> <output.xlsx>")
        sys.exit(1)

    txt_path  = sys.argv[1]
    xlsx_path = sys.argv[2]

    print(f"Parsing {txt_path} …")
    data = parse_txt(txt_path)
    print(f"Found {len(data)} unique (topology, penalty, sparsity) combinations.")

    tex_path  = xlsx_path.rsplit('.', 1)[0] + '.tex'

    wb = Workbook()
    wb.remove(wb.active)
    build_row_output(wb, data)
    build_output1(wb, data)
    wb.save(xlsx_path)
    print(f"Saved workbook to {xlsx_path}")

    build_latex(data, tex_path)
