import os
import re
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import defaultdict
from source import models
from PIL import Image, ImageColor, ImageDraw, ImageOps
import random
import pandas as pd
from plottable import ColumnDefinition, Table
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from source.utils.dataset import *

from source.utils.misc import *

def parse_experiment_folder(folder_name):
    """
    Extracts the configuration details from the folder name.
    """
    pattern = re.compile(
    r'(?P<data_code>\w+)_(?P<model>\w+)_pr(?P<pr_ratio>[0-9\.]+)_np(?P<num_partition>\d+)_(?P<sparsity_type>\w+)_(?P<experiment_flag>\w+)_(?P<reassign_flag>[\w-]+)'
)
    match = pattern.match(folder_name)
    if match:
        experiment_details = match.groupdict()
        experiment_details['run'] = experiment_details['sparsity_type'] + "_" + \
            experiment_details['experiment_flag'] + "_" + experiment_details['reassign_flag']
        words = experiment_details['sparsity_type'].split('_')
        #print(words)
        if words[0] == 'partition':
            experiment_details['topology'] = words[2]
        else:
            experiment_details['topology'] = words[1]
        return experiment_details
    return None

def load_best_accuracy(folder_path):
    """
    Reads the best accuracy value from best_accuracy.txt.
    """
    best_acc_file = os.path.join(folder_path, "best_accuracy.txt")
    if not os.path.exists(best_acc_file):
        return None
    
    with open(best_acc_file, 'r') as f:
        match = re.search(r'Best Fine-Tuned Accuracy: ([0-9\.]+)%', f.read())
        if match:
            return float(match.group(1))
    return None

def load_latest_accuracy(folder_path):
    """
    Reads the latest accuracy value from best_accuracy.txt.
    """
    best_acc_file = os.path.join(folder_path, "best_accuracy.txt")
    if not os.path.exists(best_acc_file):
        return None
    
    with open(best_acc_file, 'r') as f:
        match = re.search(r'Latest Fine-Tuned Accuracy: ([0-9\.]+)%', f.read())
        if match:
            return float(match.group(1))
    return None

def load_new_comm_cost(folder_path):

    best_acc_file = os.path.join(folder_path, "best_accuracy.txt")
    if not os.path.exists(best_acc_file):
        return None
    
    with open(best_acc_file, 'r') as f:
        match = re.search(r'Best Fine-Tuned eval_cost: ([0-9\.]+)', f.read())
        if match:
            return float(match.group(1))
    return None

def load_new_comm_cost2(folder_path):

    best_acc_file = os.path.join(folder_path, "best_accuracy.txt")
    if not os.path.exists(best_acc_file):
        return None
    
    with open(best_acc_file, 'r') as f:
        match = re.search(r'Best Fine-Tuned eval_cost_aggregate: ([0-9\.]+)', f.read())
        if match:
            return float(match.group(1))
    return None

def load_partition_data(folder_path):
    """
    Loads partition information from partition_final.
    """
    partition_file = os.path.join(folder_path, "partition_final")
    if not os.path.exists(partition_file):
        return None
    
    with open(partition_file, 'r') as f:
        partition_data = yaml.safe_load(f)
        
    # Convert lists back to NumPy arrays where applicable
    for key, value in partition_data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if (sub_key == 'filter_id') and isinstance(sub_value, list):  # Convert lists back to NumPy arrays
                    partition_data[key][sub_key] = [np.array(li, dtype=int) for li in sub_value]
    
    return partition_data

def compute_accuracy(model, dataloader, topk=(1,)):
    model.eval()
    sum_correct = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            data   = ()
            for piece in batch[:-1]:
                data += (piece.float().to(device),)
            target = batch[-1].to(device)
            data = (torch.cat(data, dim=1),)

            output = model(*data)
            total += target.size(0)

            maxk = max(topk)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.squeeze(1)  # Removes singleton dim (if topk=1, it becomes (batch_size,))
            sum_correct += pred.eq(target).sum()

    return (sum_correct / total) * 100


                
def compute_communication_cost_and_kbits(model, partition):
    """
    Computes the communication cost of the model while accounting for the fact that 
    if a machine is already computing an output filter with one input channel, 
    additional input channels for the same output filter are free.
    """
    
    if not partition or 'num' not in partition or 'maps' not in partition:
        return (0, 0)
    
    total_comm_cost = 0
    total_kbits = 0.0
    device = next(model.parameters()).device  # Ensure calculations happen on the correct device
    num_machines = partition['num']
    comm_cost_map = partition['maps']

    for name, W in model.named_parameters():
        if name in partition:
            weight = W.cpu().detach().numpy()
            shape = weight.shape
            is_conv = (len(shape) == 4)  # (out_channels, in_channels, kernel_h, kernel_w)
            
            # Retrieve partition and cost map
            layer_partition = partition[name]
            outsize = layer_partition['outsize'] if 'outsize' in layer_partition else 1
            parents = layer_partition.get('parents', [])  # Parent layers providing input
            
            if not parents:  # If no parents, skip communication cost computation
                continue
            
            for parent_layer in parents:
                if parent_layer not in partition:
                    raise ValueError(f"Parent layer {parent_layer} not found in partition dictionary.")

            # Compute the absolute sum across kernel dimensions
            if is_conv:
                W_flat = np.sum(np.abs(weight.reshape(shape[0], shape[1], -1)), axis=2)
            else:  # Fully connected layer
                W_flat = np.abs(weight)

            # Iterate through all partitions and compute communication cost
            for n in range(num_machines):  # Output filter partitions
                for C_out in layer_partition['filter_id'][n]:  # Output filters in this partition
                    for i in range(num_machines):  # Input partitions
                        if i == n:
                            continue  # Skip if input and output are on the same machine
                            
                        if len(parents) == 1:
                            input_channels = partition[parents[0]]['filter_id'][i]
                        else:
                            input_channels = np.concatenate([partition[parent]['filter_id'][i] for parent in parents])
                            input_channels = np.unique(input_channels)  # Remove duplicates

                        # Check if weights between this output filter and input channels are nonzero
                        if len(input_channels) > 0 and np.any(W_flat[C_out, input_channels]):
                            total_comm_cost += comm_cost_map[i][n] * outsize
                            partial_kbits = (outsize * 4) / 1000.0
                            total_kbits += partial_kbits

    return total_comm_cost, total_kbits

def compute_communication_loss(model, partition):
    comm_loss = 0
    device = next(model.parameters()).device
    
    for name, W in model.named_parameters():
        if name in partition:
            weight = W.cpu().detach().numpy()
            shape = weight.shape
            num_partitions = len(partition['maps'])
            cost_mask = np.zeros(shape)

            # Get the parents of this layer
            parents = partition[name].get('parents', [])

            for parent in parents:
                if parent not in partition:
                    raise ValueError(f"Parent {parent} not found in partition dict.")

                parent_filter_ids = partition[parent]['filter_id']

                for i in range(num_partitions):
                    for j in range(num_partitions):
                        if i == j:
                            continue
                        maps = partition['maps'][j][i]

                        # Use parent's filter_id instead of channel_id
                        if len(parent_filter_ids[j]) > 0 and len(partition[name]['filter_id'][i]) > 0:
                            cost_mask[partition[name]['filter_id'][i][:, None], parent_filter_ids[j]] += maps
            
            comm_cost = torch.abs(W) * torch.from_numpy(cost_mask).to(device)
            comm_cost = comm_cost.view(comm_cost.size(0), -1).sum()
            if partition[name]['outsize']:
                comm_loss += comm_cost*partition[name]['outsize']
            else:
                comm_loss += comm_cost

    return comm_loss.item()

def compute_model_sparsity(model, partition, mode="kernel"):
    """
    Returns a float between 0 and 1, the fraction of zeros (or zero-structured 
    groups) under a chosen 'mode'.

    Options:
      - mode="parameter": fraction of individual parameters that are zero
      - mode="kernel": fraction of (outC,inC) kernels that are entirely zero
      - mode="out_channel": fraction of output channels that are entirely zero
      - mode="in_channel": fraction of input channels that are entirely zero
      ...
    Adjust or extend as desired.
    """
    # If the user wants something else, adapt. For now we demonstrate these four:
    zero_count = 0
    total_count = 0

    for name, W in model.named_parameters():
        if name not in partition:
            continue
        weight = W.detach().cpu().numpy()
        shape = weight.shape
        
        # "parameter" => standard elementwise
        if mode == "parameter":
            total_params = weight.size
            zeros = np.count_nonzero(weight == 0)
            zero_count += zeros
            total_count += total_params

        # "kernel" => in conv2D, each (outC, inC) is a small kernel 
        # if *all elements* in that kernel are zero => kernel is zero
        elif mode == "kernel":
            if len(shape) == 4:
                # shape: (outC, inC, kH, kW)
                # Flatten the last two dims => (outC, inC, kH*kW)
                # Check if all are zero
                w2 = weight.reshape(shape[0], shape[1], -1)
                total_kernels = shape[0]*shape[1]
                kernel_is_zero = np.all(w2 == 0, axis=2)  # shape (outC, inC), True if entire kernel is zero
                zero_kernels = np.count_nonzero(kernel_is_zero)
                zero_count += zero_kernels
                total_count += total_kernels
            elif len(shape) == 2:
                # For linear, interpret each (outC, inC) as a "kernel"
                # It's zero if all elements in row col? That row col is just a single element? 
                # Actually we can treat each (i,j) as a "kernel" of size 1
                # => fallback to parameter if you prefer
                total_params = weight.size
                zeros = np.count_nonzero(weight == 0)
                zero_count += zeros
                total_count += total_params

        # "out_channel" => entire out_channel is zero if all elements in that channel are zero
        elif mode == "out_channel":
            if len(shape) == 4:
                # shape: (outC, inC, kH, kW)
                w2 = weight.reshape(shape[0], -1)  # flatten inC,kH,kW
                total_outC = shape[0]
                outC_zero = np.all(w2 == 0, axis=1)
                zero_count += np.count_nonzero(outC_zero)
                total_count += total_outC
            elif len(shape) == 2:
                # shape: (out_dim, in_dim)
                total_outC = shape[0]
                outC_zero = np.all(weight == 0, axis=1)
                zero_count += np.count_nonzero(outC_zero)
                total_count += total_outC

        # "in_channel" => entire in_channel is zero if all elements in that column are zero
        elif mode == "in_channel":
            if len(shape) == 4:
                # shape: (outC, inC, kH, kW)
                w2 = weight.transpose(1,0,2,3).reshape(shape[1], -1)  # gather all outC,kH,kW for each inC
                total_inC = shape[1]
                inC_zero = np.all(w2 == 0, axis=1)
                zero_count += np.count_nonzero(inC_zero)
                total_count += total_inC
            elif len(shape) == 2:
                # shape: (out_dim, in_dim)
                total_inC = shape[1]
                inC_zero = np.all(weight == 0, axis=0)
                zero_count += np.count_nonzero(inC_zero)
                total_count += total_inC
        
        elif mode == "partition_row":
            num_partitions = partition['num']
            if len(shape) != 4:
                continue  # Only handle Conv layers for now
            weight3d = weight.reshape(shape[0], shape[1], -1)

            for dst in range(num_partitions):
                output_filters = partition[name]['filter_id'][dst]
                for src in range(num_partitions):
                    if src == dst:
                        continue
                    if len(parents) == 1:
                        input_channels = partition[parents[0]]['filter_id'][src]
                    else:
                        input_channels = np.concatenate([partition[parent]['filter_id'][src] for parent in parents])
                        input_channels = np.unique(input_channels)

                    if len(output_filters) == 0 or len(input_channels) == 0:
                        continue

                    submatrix = weight3d[np.ix_(output_filters, input_channels)]
                    norms = LA.norm(submatrix.reshape(submatrix.shape[0], -1), axis=1)
                    zero_count += np.count_nonzero(norms == 0)
                    total_count += len(norms)
        
        else:
            # fallback: treat as "parameter"
            total_params = weight.size
            zeros = np.count_nonzero(weight == 0)
            zero_count += zeros
            total_count += total_params

    if total_count == 0:
        return 0.0
    return zero_count / total_count


def compute_roc_auc(model, test_loader, device='cpu'):
    model = model.to(device)
    
    model.eval()
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            control=False   # True: run for smaller dataset
            if batch_idx > 5 and control:
                print("!!!! Running for SMALL dataset !!!!")
                break
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            y_score.extend(probs.cpu().numpy())  # Untuk ROC AUC
            y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())  # Pred label
            y_true.extend(labels.cpu().numpy())  # Label ground truth

    # Convert ke array numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # ROC AUC Score (multi-class, macro average)
    roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
    acc = accuracy_score(y_true, y_pred)
    print(f"\nMulti-class ROC AUC Score (macro): {roc_auc:.4f}, Accuracy: {acc:.4f}")

    '''
    prec, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print(f"\nPrecision (macro): {prec:.4f}")
    print(f"\nRecall (macro): {recall:.4f}")
    print(f"\nF1 Score (macro): {fscore:.4f}")
    '''
    return round(roc_auc,3), round(acc,3)
    
def plot_layer(model, partition, layer_ids, save_folder="layer_vis", key="", rsgn=""):
    """
    Generates **side-by-side visualizations** of layer connectivity:
    1) **Ordered by Output Channels** (rows reordered using `filter_id`).
    2) **Ordered by Parent Input Channels** (columns reordered using parent `filter_id`).

    - **Color Encoding:**
        - **White (0):** No connection
        - **Red (1):** Cross-partition communication
        - **Green (2):** Active connection within partition

    - **File Naming:** `{experiment_flag}_{rsgn}_{layer_name}_from_{parent1}_{parent2}.png`
    """

    os.makedirs(save_folder, exist_ok=True)

    def reorder_rows(mat, filter_id):
        """Reorders rows based on `filter_id` ordering."""
        new_order = np.concatenate(filter_id).astype(int)
        return mat[new_order, :] if len(new_order) > 0 else mat

    def reorder_cols(mat, filter_id):
        """Reorders columns based on `filter_id` ordering."""
        new_order = np.concatenate(filter_id).astype(int)
        return mat[:, new_order] if len(new_order) > 0 else mat

    def color_matrix(base_mat, part, parent):
        """Encodes matrix values for visualization (2=green, 1=red, 0=white)."""
        base_mat[base_mat!=0] = 2 #Set all non-zero weights to green
        for i in range(part['num']):
            if parent == "inputs":
                continue  # Skip input dependencies

            if parent not in part:
                raise ValueError(f"Parent layer {parent} not found in part dictionary.")
            parent_filter_ids = part[parent]['filter_id']
            for j in range(part['num']):
                if i==j: continue
                if len(part[name]['filter_id'][i]) > 0 and len(parent_filter_ids[j]) > 0:
                    rows = part[name]['filter_id'][i]
                    cols = parent_filter_ids[j]
                    if len(rows) > 0 and len(cols) > 0:
                        rr, cc = np.meshgrid(rows, cols, indexing='ij')
                        mask = base_mat[rr, cc] != 0
                        base_mat[rr[mask], cc[mask]] = 1
        return base_mat

    def matrix_to_image(mat, row_partitions=None, col_partitions=None, line_color="black"):
        """Converts matrix into color-coded image (white, red, green)."""
        red, green = ImageColor.getcolor("#FF0606", "RGB"), ImageColor.getcolor("#0FA958", "RGB")
        height, width = mat.shape
        img = Image.new("RGB", (width, height), "white")
        pixels = img.load()
        
        for j in range(mat.shape[0]):
            for i in range(mat.shape[1]):
                pixels[i, j] = ((255, 255, 255) if mat[j, i] == 0 else (red if mat[j, i] == 1 else green))
        
        draw = ImageDraw.Draw(img)
        if col_partitions:
            for col in col_partitions:
                if 0 < col < width:
                    draw.line([(col, 0), (col, height)], fill=line_color)
        if row_partitions:
            for row in row_partitions:
                if 0 < row < height:
                    draw.line([(0, row), (width, row)], fill=line_color)

        scale = 2  # or 10, depending on how big you want it
        img = img.resize((width * scale, height * scale), resample=Image.NEAREST)
        return ImageOps.expand(img, border=1, fill='black')
    
    def get_partition_boundaries(filter_id_list):
        """
        Given an ordered dict (or normal dict where iteration order matches final layout),
        returns partition boundaries (cumulative lengths).

        Each value in filter_id_dict should be a list of indices *in the new reordered matrix*.
        """
        boundaries = []
        offset = 0
        for ids in filter_id_list:
            offset += len(ids)
            if offset > 0:
                boundaries.append(offset)
        return boundaries[:-1]  # exclude final edge
    
    counter = 0
    for name, W in model.named_parameters():
        if name not in partition:
            continue
        counter += 1
        if counter not in layer_ids:
            continue
        weight = W.detach().cpu().numpy()
        weight2d = np.abs(weight.reshape(weight.shape[0], weight.shape[1], -1)).sum(-1)  # Flatten and add along kernel dims
        # Color red inter-partition cells
        parents = partition[name].get('parents', [])
        sub_images = []
        
        for parent in parents:
            if parent in partition:
                weight2d_col = color_matrix(weight2d, partition, parent)
                 # Reorder by Output Channels and Input Channels
                weight2d_out = reorder_rows(weight2d_col, partition[name]['filter_id'])
                row_lines = get_partition_boundaries(partition[name]['filter_id'])
                weight2d_in = reorder_cols(weight2d_out, partition[parent]['filter_id'])
                col_lines = get_partition_boundaries(partition[parent]['filter_id'])
                sub_images.append(matrix_to_image(weight2d_in, row_partitions=row_lines, col_partitions=col_lines))

        # **Combine Images Horizontally**
        total_width = sum(im.width for im in sub_images)
        max_height = max(im.height for im in sub_images)
        combined = Image.new("RGB", (total_width, max_height), "white")
        x_offset = 0
        for im in sub_images:
            combined.paste(im, (x_offset, 0))
            x_offset += im.width

        # **Save Image**
        parents_str = "_".join(parents) if parents else "noParent"
        key_string = "_".join(key[:])
        out_fname = f"{key_string}_{rsgn}_{name.replace('.', '_')}_from_{parents_str}.png"
        combined.save(os.path.join(save_folder, out_fname))
        print(f"Saved: {out_fname}")

def to_float(x):
    return float(x.detach().numpy()) if isinstance(x, torch.Tensor) else float(x)  

def extend_non_complete_runs(exp, runs_0):
    for key, item in exp.items():
        for idx, (reassign_flag, runs) in enumerate(item.items()):
            if len(runs) == 3:
                # This means we have not completed the 0 run for this model
                rsgn = 'rsgn' if 'rsgn' in reassign_flag else 'fixed'
                exp[key][reassign_flag].append(runs_0[key[3]][rsgn])
    return exp

# Refer to https://github.com/znstrider/plottable/blob/master/docs/example_notebooks/wwc_example.ipynb
def plot_table_metrics(table_experiments, output_dir="."):


    datacodes = []
    models = []
    topologies = []
    partitions = []
    exp_runs = []
    prune_ratios = []
    accuracies = []
    latest_accuracies = []
    comm_costs = []
    comm_losses = []
    eval_costs = []
    eval_cost_aggregates = []
    kernel_sparcities = []
    kbs = []
    aucs = []

    for key, reassignments in table_experiments.items():
        #print("KEYYYYY: ", key)
        datacode = key[0]
        model = key[1]
        partition = key[2]
        topology = key[3]
        #print(partition, penalty)
        for idx, (reassign_flag, runs) in enumerate(reassignments.items()):
            #print("reassign_flag: ", reassign_flag)
        
            for run in runs:
                #print("-----------------")
                #print(run)
                partitions.append(partition)
                models.append(model)
                datacodes.append(datacode)
                exp_runs.append(reassign_flag)
                prune_ratios.append(run[0])
                accuracies.append(run[1])
                comm_costs.append(run[2])
                comm_losses.append(round(run[3],1))
                eval_costs.append(run[6])
                eval_cost_aggregates.append(run[7])
                kernel_sparcities.append(round(run[5],2))
                kbs.append(round(run[4],1))  # total_kbits
                latest_accuracies.append(run[8])
                aucs.append(run[9])
                topologies.append(topology)   
    '''
    print('Penalty: ', penalties)
    print('Partitions: ', partitions)
    print('Topology: ', topologies)
    print('Run: ', exp_runs)
    print('Prune Ratio: ', prune_ratios)
    print('Accuracy: ', accuracies)
    print('Kernel sparcity: ', kernel_sparcities)
    print('Comm Cost: ', comm_costs)
    print('Eval Cost: ', eval_costs)
    print('Aggregate Cost:', eval_cost_aggregates)
    '''

    dict = {'Dataset': datacodes, 'Model': models, 'Topology': topologies, 'Partitions': partitions, 
        'Run': exp_runs, 'Prune Ratio': prune_ratios, 'Best Accuracy': accuracies, 'Last Accuracy': latest_accuracies,
        'Comm Cost': comm_costs, 'Eval Cost': eval_costs, 'Aggregate Cost': eval_cost_aggregates, 'Comm Cost(MB)': kbs,
        'Comm Loss': comm_losses, 'Sparcity': kernel_sparcities, 'AUC': aucs} 

    df = pd.DataFrame(dict)
    save_fname = os.path.join(output_dir, f"experiments_metrics.csv")
    df.to_csv(save_fname, index=False)
    
    df = df.set_index("Dataset")
      
    #print(df)  

    colnames = [
        "Dataset",
        "Model"
        "Topology",
        "Partitions",
        "Run",
        "Prune Ratio",
        "Best Accuracy",
        "Last Accuracy",
        "Comm Cost",
        "Eval Cost",
        "Aggregate Cost",
        "Comm Cost(MB)",
        "Comm Loss",
        "Sparcity",
        "AUC"

    ]

    col_defs = (
    [
        ColumnDefinition(
            name="Dataset",
            textprops={"ha": "left"},
            width=0.35,
        ),
        ColumnDefinition(
            name="Model",
            textprops={"ha": "left"},
            width=0.35,
        ),
        ColumnDefinition(
            name="Topology",
            #group="Team Rating",
            textprops={"ha": "left"},
            width=0.35,
        ),
        ColumnDefinition(
            name="Partitions",
            textprops={"ha": "center"},
            width=0.35,
        ),
        ColumnDefinition(
            name="Run",
            textprops={"ha": "left"},
            width=2.5,
        ),
        ColumnDefinition(
            name="Prune Ratio",
            textprops={"ha": "center"},
            width=0.35,
        ),
        ColumnDefinition(
            name="Best Accuracy",
            #group="Team Rating",
            textprops={"ha": "center"},
            width=0.35,
        ),
        ColumnDefinition(
            name="Last Accuracy",
            #group="Team Rating",
            textprops={"ha": "center"},
            width=0.35,
        ),
        ColumnDefinition(
            name="Comm Cost",
            textprops={"ha": "left"},
            width=0.35,
        ),
        ColumnDefinition(
            name="Aggregate Cost",
            textprops={"ha": "center"},
            width=0.35,
        ),
        ColumnDefinition(
            name="Eval Cost",
            #group="Team Rating",
            textprops={"ha": "center"},
            width=0.35,
        ),
        ColumnDefinition(
            name="Comm Cost(MB)",
            #group="Team Rating",
            textprops={"ha": "center"},
            width=0.35,
        ),
        ColumnDefinition(
            name="Comm Loss",
            #group="Team Rating",
            textprops={"ha": "center"},
            width=0.35,
        ),
        ColumnDefinition(
            name="Sparcity",
            #group="Team Rating",
            textprops={"ha": "center"},
            width=0.35,
        ),
        ColumnDefinition(
            name="AUC",
            #group="Team Rating",
            textprops={"ha": "center"},
            width=0.35,
        ),
    ])

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["savefig.bbox"] = "tight"

    fig, ax = plt.subplots(figsize=(30, 20))

    fig.suptitle(f"Experiments' Metrics", fontsize=10, y=0.9)

    table = Table(
        df,
        column_definitions=col_defs,
        #row_dividers=True,
        #footer_divider=True,
        ax=ax,
        textprops={"fontsize": 10},
        #row_divider_kw={"linewidth": 0.5, "linestyle": (0, (1, 5))},
        #col_label_divider_kw={"linewidth": 0.5, "linestyle": "-"},
        #column_border_kw={"linewidth": 0.5, "linestyle": "-"},
    )#.autoset_fontcolors(colnames=["OFF", "DEF"])

    save_fname = os.path.join(
        output_dir,
        f"experiments_metrics.png"
    )
    fig.savefig(save_fname, facecolor=ax.get_facecolor(), dpi=200)
    return 

# Prun ratios 1.0 and 0.0 are plotted as DOTs !!!
def new_plot_all_metrics(experiments, output_dir=".", sparsity_mode="kernel", mode="ACC"):
    pd.options.display.max_colwidth = 100
    # Custom color cycle for each reassign_flag line
    colors = ["#FFC20A", "#0C7BDC", "#D41159", "#004D40", "#45D86E", "#3E2717", "#FE7800"]
    markers = ['.', '*', 'v', 'o']
    linestyles = ['-', '--', '-.', ':']

    all_runs = []
    accs_pr_one = []
    accs_pr_zero = []
    costs_pr_one = []
    costs_pr_zero = []
    loss_pr_one = []
    loss_pr_zero = []
    kbits_pr_one = []
    kbits_pr_zero = []

    for key, reassignments in experiments.items():
        
        key_string = "_".join(key[:])
        
        # Create a new 2x2 figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

        # axes[0,0] => Accuracy vs. Comm. Cost
        # axes[0,1] => Accuracy vs. Comm. Loss
        # axes[1,0] => Accuracy vs. Kbits
        # axes[1,1] => Sparsity vs. Kbits
        
        # For the cost, we use a “free-channels” logic expression, e.g.:
        #   Cost(W,M) = sum_{l=1 to L} sum_{j=1}^{n_l} sum_{p != M(j)} c_{p,M(j)}
        #        * 1{there exists i: W_l(i,j) != 0 and M(i)=p}
        #
        # For the loss, we sum over edges with absolute weights:
        #   Loss(W,M) = sum_{l=1}^L sum_{(i,j) in E_l} c_{M(i), M(j)} * |W_l(i,j)|

        # Convert axes to friendly names
        ax_cost  = axes[0,0]
        ax_loss  = axes[0,1]
        ax_kbits = axes[1,0]
        ax_spars = axes[1,1]
        
        legend_entries = []
        
        for idx, (reassign_flag, runs) in enumerate(reassignments.items()):
            marker = markers[0] if 'fixed' in reassign_flag else markers[1]
            # If you want to add a highlight line
            marker = markers[2] if 'prune_comm' in reassign_flag else marker
            linestyle = linestyles[1] if 'partition_row' in reassign_flag else linestyles[0]
            #print(reassign_flag, runs)
            runs_sorted = sorted(runs, key=lambda x: x[0])  # sort by prune_ratio
            if reassign_flag.find("kernel") == -1:
                # Does NOT contain Kernel
                all_runs.append(reassign_flag)
                #print("====", reassign_flag)
                #print("zzz", runs_sorted[0][0])
                #print("ooo", runs_sorted[-1][0])
                if runs_sorted[0][0] == 0.0:
                    #print("XXXXX: ", runs_sorted[0][0], runs_sorted[0][1])
                    if mode == "ACC":
                        accs_pr_zero.append(runs_sorted[0][1]) # accuracy
                    elif mode == "AUC":
                        accs_pr_zero.append(runs_sorted[0][7])  # roc_auc
                    costs_pr_zero.append(runs_sorted[0][2])
                    loss_pr_zero.append(runs_sorted[0][3])
                    kbits_pr_zero.append(runs_sorted[0][4])

                    pr_ratio_zero = runs_sorted.pop(0)
                    
                if runs_sorted[-1][0] == 1.0:
                    #print("XXXXX: ", runs_sorted[-1][0], runs_sorted[-1][1])
                    if mode == "ACC":
                        accs_pr_one.append(runs_sorted[-1][1])  # accuracy
                    elif mode == "AUC":
                        accs_pr_one.append(runs_sorted[-1][7])  # roc_auc
                    costs_pr_one.append(runs_sorted[-1][2])
                    loss_pr_one.append(runs_sorted[-1][3])
                    kbits_pr_one.append(runs_sorted[-1][4])

                    pr_ratio_one = runs_sorted.pop(-1)
                    
                #print("final: runs_sorted: ", runs_sorted)
            pr_vals, acc_vals, cost_vals, loss_vals, kbits_vals, spar_vals, latest_acc_vals, roc_auc_vals = map(lambda vals: [to_float(v) for v in vals], zip(*runs_sorted))

            # Pick a color from the custom list
            color = colors[idx % len(colors)]
            legend_entries.append((linestyle, marker, color, reassign_flag))

            if mode == "ACC":
                y_values = acc_vals
                inline_text = "Accuracy"
                ylabel_text = "Accuracy (%)"
            elif mode == "AUC":
                inline_text = "AUC"
                ylabel_text = "AUC"
                y_values = roc_auc_vals

            # 1) Accuracy vs. Comm. Cost
            ax_cost.plot(cost_vals, y_values, marker=marker, markersize=12, linestyle=linestyle, 
                            label=f"{reassign_flag}", color=color)
            for i, prr in enumerate(pr_vals):
                ax_cost.text(cost_vals[i], y_values[i], f"pr={prr}", fontsize=10)
            
            # 2) Accuracy vs. Comm. Loss
            ax_loss.plot(loss_vals, y_values, marker=marker, markersize=12, linestyle=linestyle, 
                            label=f"{reassign_flag}", color=color)
            for i, prr in enumerate(pr_vals):
                ax_loss.text(loss_vals[i], y_values[i], f"pr={prr}", fontsize=10)

            # 3) Accuracy vs. Kbits
            ax_kbits.plot(kbits_vals, y_values, marker=marker, markersize=12, linestyle=linestyle, 
                            label=f"{reassign_flag}", color=color)
            for i, prr in enumerate(pr_vals):
                ax_kbits.text(kbits_vals[i], y_values[i], f"pr={prr}", fontsize=10)

            # 4) Sparsity vs. Kbits
            ax_spars.plot(kbits_vals, spar_vals, marker=marker, markersize=12, linestyle=linestyle, 
                            label=f"{reassign_flag}", color=color)
            for i, prr in enumerate(pr_vals):
                ax_spars.text(kbits_vals[i], spar_vals[i], f"pr={prr}", fontsize=10)

        # ----------------------------------------------------------------
        # Titles with LaTeX formulas for Comm. Cost & Comm. Loss
        # ----------------------------------------------------------------

        # Top-left: Comm. Cost formula
        ax_cost.set_title(
            inline_text + " vs. Comm. Cost\n" +
            r"$\mathrm{Cost}(\mathbf{W},\mathbf{M}) = "
            r"\sum_{l=1}^{L}\sum_{j=1}^{n_l}\sum_{p\neq \mathbf{M}(j)}"
            r" c_{p,\mathbf{M}(j)} \cdot 1_{\{\exists i : W_l(i,j)\neq 0,\,M(i)=p\}}$"
        )

        ax_loss.set_title(
            inline_text + " vs. Comm. Loss\n" +
            r"$\mathrm{Loss}(\mathbf{W},\mathbf{M}) = "
            r"\sum_{l=1}^{L}\sum_{(i,j)\in E_l} "
            r"c_{M(i), M(j)} \cdot |W_l(i,j)|$"
        )

        # Bottom-left: Accuracy vs Kbits
        ax_kbits.set_title(inline_text + " vs. Kbits Transmitted")
        # Bottom-right: Sparsity vs Kbits
        ax_spars.set_title(f"{sparsity_mode.capitalize()} Sparsity vs. Kbits")

        # ----------------------------------------------------------------
        # Axis labels, grid, legend, etc.
        # ----------------------------------------------------------------

        # Top-left
        ax_cost.set_xlabel("Comm. Cost")
        ax_cost.set_ylabel(ylabel_text)
        ax_cost.grid(True)
        #ax_cost.legend()

        # Top-right
        ax_loss.set_xlabel("Comm. Loss")
        ax_loss.set_ylabel(ylabel_text)
        ax_loss.grid(True)
        #ax_loss.legend()

        # Bottom-left
        ax_kbits.set_xlabel("Kbits Transmitted")
        ax_kbits.set_ylabel(ylabel_text)
        ax_kbits.grid(True)
        #ax_kbits.legend()

        # Bottom-right
        ax_spars.set_xlabel("Kbits Transmitted")
        ax_spars.set_ylabel(f"{sparsity_mode.capitalize()} Sparsity")
        ax_spars.grid(True)
        #ax_spars.legend()


        #print(len(all_runs))
        #print(accs_pr_one) 
        #print(accs_pr_zero)  

        data = {'runs': all_runs,
                'accs_pr_zero': accs_pr_zero,
                'accs_pr_one': accs_pr_one,
                'costs_pr_zero': costs_pr_zero,
                'costs_pr_one': costs_pr_one,
                'loss_pr_zero': loss_pr_zero,
                'loss_pr_one': loss_pr_one,
                'kbits_pr_zero': kbits_pr_zero,
                'kbits_pr_one': kbits_pr_one,
                } 
        df = pd.DataFrame(data)
        max_zero_idx = df['accs_pr_zero'].idxmax()
        max_one_idx = df['accs_pr_one'].idxmax()
        #print(df)
        #print(max_zero_idx, max_one_idx)
        acc_vs_comm_cost_one = (df.loc[max_one_idx, 'runs'], df.loc[max_one_idx, 'costs_pr_one'], df.loc[max_one_idx, 'accs_pr_one'] )
        acc_vs_comm_cost_zero = (df.loc[max_zero_idx, 'runs'], df.loc[max_zero_idx, 'costs_pr_zero'], df.loc[max_zero_idx, 'accs_pr_zero'])
        acc_vs_comm_loss_one = (df.loc[max_one_idx, 'runs'], df.loc[max_one_idx, 'loss_pr_one'], df.loc[max_one_idx, 'accs_pr_one'] )
        acc_vs_comm_loss_zero = (df.loc[max_zero_idx, 'runs'], df.loc[max_zero_idx, 'loss_pr_zero'], df.loc[max_zero_idx, 'accs_pr_zero'])
        acc_vs_comm_kbits_one = (df.loc[max_one_idx, 'runs'], df.loc[max_one_idx, 'kbits_pr_one'], df.loc[max_one_idx, 'accs_pr_one'] )
        acc_vs_comm_kbits_zero = (df.loc[max_zero_idx, 'runs'], df.loc[max_zero_idx, 'kbits_pr_zero'], df.loc[max_zero_idx, 'accs_pr_zero'])
        #print(acc_vs_comm_cost_one)
        #print(acc_vs_comm_cost_zero)

        # Add prune ratios 0.0 and 1.0 at Accuracy vs. Comm. Cost plot
        # 'ro': red dot, 'bs': blue square
        ax_cost.plot(acc_vs_comm_cost_one[1], acc_vs_comm_cost_one[2], 'ro', markersize=10, label='Square dots')
        ax_cost.plot(acc_vs_comm_cost_zero[1], acc_vs_comm_cost_zero[2], 'bs', markersize=10, label='Square dots')
        ax_cost.text(acc_vs_comm_cost_one[1], acc_vs_comm_cost_one[2], f"pr=1.0", fontsize=10)
        ax_cost.text(acc_vs_comm_cost_zero[1], acc_vs_comm_cost_zero[2], f"pr=0.0", fontsize=10)

        # Add prune ratios 0.0 and 1.0 at Accuracy vs. Comm. Loss plot
        # 'ro': red dot, 'bs': blue square
        ax_loss.plot(acc_vs_comm_loss_one[1], acc_vs_comm_loss_one[2], 'ro', markersize=10, label='Square dots')
        ax_loss.plot(acc_vs_comm_loss_zero[1], acc_vs_comm_loss_zero[2], 'bs', markersize=10, label='Square dots')
        ax_loss.text(acc_vs_comm_loss_one[1], acc_vs_comm_loss_one[2], f"pr=1.0", fontsize=10)
        ax_loss.text(acc_vs_comm_loss_zero[1], acc_vs_comm_loss_zero[2], f"pr=0.0", fontsize=10)

        # Add prune ratios 0.0 and 1.0 at Accuracy vs. Kbits plot
        # 'ro': red dot, 'bs': blue square
        ax_kbits.plot(acc_vs_comm_kbits_one[1], acc_vs_comm_kbits_one[2], 'ro', markersize=10, label='Square dots')
        ax_kbits.plot(acc_vs_comm_kbits_zero[1], acc_vs_comm_kbits_zero[2], 'bs', markersize=10, label='Square dots')
        ax_kbits.text(acc_vs_comm_kbits_one[1], acc_vs_comm_kbits_one[2], f"pr=1.0", fontsize=10)
        ax_kbits.text(acc_vs_comm_kbits_zero[1], acc_vs_comm_kbits_zero[2], f"pr=0.0", fontsize=10)


        legend_entries.append(('', 'o', "red", acc_vs_comm_cost_one[0]))
        legend_entries.append(('', 's', "blue", acc_vs_comm_cost_zero[0]))
        

        # Create a single legend further below the plots
        #print(legend_entries)
        handles = [plt.Line2D([0], [0], linestyle=style, marker=mark, color=col, markersize=10, label=lbl) 
                    for style, mark, col, lbl in legend_entries]
        fig.legend(handles=handles, loc='lower center', fontsize=14, ncol=4, bbox_to_anchor=(0.5, -0.08))

        # Optional figure-level title referencing `key`
        fig.suptitle(f"All Metrics for {key_string}", fontsize=14, y=0.9)

        plt.tight_layout(rect=[0, 0, 1, 0.90])  # leaves space for suptitle

        # Save the figure
        if mode == "ACC":
            save_fname = os.path.join(output_dir, f"{key_string}_dot_metrics.png")
        elif mode == "AUC":
            save_fname = os.path.join(output_dir, f"{key_string}_auc_dot_metrics.png")
        plt.savefig(save_fname, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to {save_fname}")



# Valid values for mode: ACC, AUC        
def plot_all_metrics(experiments, output_dir=".", sparsity_mode="kernel", mode="ACC"):
    """
    We assume 'experiments' is a dict:
        experiments = {
          key: {   # e.g. (data_code, model, np, exp_flag)
            reassign_flag: [
              (prune_ratio, best_acc, comm_cost, comm_loss, total_kbits, model_sparsity),
              ...
            ],
            ...
          },
          ...
        }
    
    Produces a single 2x2 figure for each `key`:
      (1) Accuracy vs. Comm. Cost   (top-left)
      (2) Accuracy vs. Comm. Loss   (top-right)
      (3) Accuracy vs. Kbits        (bottom-left)
      (4) Sparsity vs. Kbits        (bottom-right)
      
    The formulas added in the subplot titles are:

      Comm. Cost:
        Cost(W,M) = sum_{l=1}^L sum_{j=1}^{n_l} sum_{p != M(j)} c_{p,M(j)} * 1{exists i : W_l(i,j) != 0 and M(i)=p}

      Comm. Loss:
        Loss(W,M) = sum_{l=1}^L sum_{(i,j) in E_l} c_{M(i), M(j)} * |W_l(i,j)|

    Each line is plotted in a color from your 'colors' list, based on the index of reassign_flag.
    """
    
    # Custom color cycle for each reassign_flag line
    colors = ["#FFC20A", "#0C7BDC", "#D41159", "#004D40", "#45D86E", "#3E2717", "#FE7800"]
    markers = ['.', '*', 'v']
    linestyles = ['-', '--', '-.', ':']
    
    for key, reassignments in experiments.items():
        
        key_string = "_".join(key[:])
        
        # Create a new 2x2 figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

        # axes[0,0] => Accuracy vs. Comm. Cost
        # axes[0,1] => Accuracy vs. Comm. Loss
        # axes[1,0] => Accuracy vs. Kbits
        # axes[1,1] => Sparsity vs. Kbits
        
        # For the cost, we use a “free-channels” logic expression, e.g.:
        #   Cost(W,M) = sum_{l=1 to L} sum_{j=1}^{n_l} sum_{p != M(j)} c_{p,M(j)}
        #        * 1{there exists i: W_l(i,j) != 0 and M(i)=p}
        #
        # For the loss, we sum over edges with absolute weights:
        #   Loss(W,M) = sum_{l=1}^L sum_{(i,j) in E_l} c_{M(i), M(j)} * |W_l(i,j)|

        # Convert axes to friendly names
        ax_cost  = axes[0,0]
        ax_loss  = axes[0,1]
        ax_kbits = axes[1,0]
        ax_spars = axes[1,1]
        
        legend_entries = []
        
        for idx, (reassign_flag, runs) in enumerate(reassignments.items()):
            marker = markers[0] if 'fixed' in reassign_flag else markers[1]
            # If you want to add a highlight line
            marker = markers[2] if 'prune_comm' in reassign_flag else marker
            linestyle = linestyles[1] if 'partition_row' in reassign_flag else linestyles[0]
            #print(reassign_flag, runs)
            runs_sorted = sorted(runs, key=lambda x: x[0])  # sort by prune_ratio
            pr_vals, acc_vals, cost_vals, loss_vals, kbits_vals, spar_vals, latest_acc_vals, roc_auc_vals = map(lambda vals: [to_float(v) for v in vals], zip(*runs_sorted))

            # Pick a color from the custom list
            color = colors[idx % len(colors)]
            legend_entries.append((linestyle, marker, color, reassign_flag))

            if mode == "ACC":
                y_values = acc_vals
                inline_text = "Accuracy"
                ylabel_text = "Accuracy (%)"
            elif mode == "AUC":
                inline_text = "AUC"
                ylabel_text = "AUC"
                y_values = roc_auc_vals

            # 1) Accuracy vs. Comm. Cost
            ax_cost.plot(cost_vals, y_values, marker=marker, markersize=12, linestyle=linestyle, 
                         label=f"{reassign_flag}", color=color)
            for i, prr in enumerate(pr_vals):
                ax_cost.text(cost_vals[i], y_values[i], f"pr={prr}", fontsize=10)

            # 2) Accuracy vs. Comm. Loss
            ax_loss.plot(loss_vals, y_values, marker=marker, markersize=12, linestyle=linestyle, 
                         label=f"{reassign_flag}", color=color)
            for i, prr in enumerate(pr_vals):
                ax_loss.text(loss_vals[i], y_values[i], f"pr={prr}", fontsize=10)

            # 3) Accuracy vs. Kbits
            ax_kbits.plot(kbits_vals, y_values, marker=marker, markersize=12, linestyle=linestyle, 
                         label=f"{reassign_flag}", color=color)
            for i, prr in enumerate(pr_vals):
                ax_kbits.text(kbits_vals[i], y_values[i], f"pr={prr}", fontsize=10)

            # 4) Sparsity vs. Kbits
            ax_spars.plot(kbits_vals, spar_vals, marker=marker, markersize=12, linestyle=linestyle, 
                         label=f"{reassign_flag}", color=color)
            for i, prr in enumerate(pr_vals):
                ax_spars.text(kbits_vals[i], spar_vals[i], f"pr={prr}", fontsize=10)

        # ----------------------------------------------------------------
        # Titles with LaTeX formulas for Comm. Cost & Comm. Loss
        # ----------------------------------------------------------------

        # Top-left: Comm. Cost formula
        ax_cost.set_title(
            inline_text + " vs. Comm. Cost\n" +
            r"$\mathrm{Cost}(\mathbf{W},\mathbf{M}) = "
            r"\sum_{l=1}^{L}\sum_{j=1}^{n_l}\sum_{p\neq \mathbf{M}(j)}"
            r" c_{p,\mathbf{M}(j)} \cdot 1_{\{\exists i : W_l(i,j)\neq 0,\,M(i)=p\}}$"
        )

        ax_loss.set_title(
            inline_text + " vs. Comm. Loss\n" +
            r"$\mathrm{Loss}(\mathbf{W},\mathbf{M}) = "
            r"\sum_{l=1}^{L}\sum_{(i,j)\in E_l} "
            r"c_{M(i), M(j)} \cdot |W_l(i,j)|$"
        )

        # Bottom-left: Accuracy vs Kbits
        ax_kbits.set_title(inline_text + " vs. Kbits Transmitted")
        # Bottom-right: Sparsity vs Kbits
        ax_spars.set_title(f"{sparsity_mode.capitalize()} Sparsity vs. Kbits")

        # ----------------------------------------------------------------
        # Axis labels, grid, legend, etc.
        # ----------------------------------------------------------------

        # Top-left
        ax_cost.set_xlabel("Comm. Cost")
        ax_cost.set_ylabel(ylabel_text)
        ax_cost.grid(True)
        #ax_cost.legend()

        # Top-right
        ax_loss.set_xlabel("Comm. Loss")
        ax_loss.set_ylabel(ylabel_text)
        ax_loss.grid(True)
        #ax_loss.legend()

        # Bottom-left
        ax_kbits.set_xlabel("Kbits Transmitted")
        ax_kbits.set_ylabel(ylabel_text)
        ax_kbits.grid(True)
        #ax_kbits.legend()

        # Bottom-right
        ax_spars.set_xlabel("Kbits Transmitted")
        ax_spars.set_ylabel(f"{sparsity_mode.capitalize()} Sparsity")
        ax_spars.grid(True)
        #ax_spars.legend()
        
        # Create a single legend further below the plots
        #print(legend_entries)
        handles = [plt.Line2D([0], [0], linestyle=style, marker=mark, color=col, markersize=10, label=lbl) 
                   for style, mark, col, lbl in legend_entries]
        fig.legend(handles=handles, loc='lower center', fontsize=14, ncol=4, bbox_to_anchor=(0.5, -0.08))
        
        # Optional figure-level title referencing `key`
        fig.suptitle(f"All Metrics for {key_string}", fontsize=14, y=0.9)

        plt.tight_layout(rect=[0, 0, 1, 0.90])  # leaves space for suptitle

        # Save the figure
        if mode == "ACC":
            save_fname = os.path.join(output_dir, f"{key_string}_metrics.png")
        elif mode == "AUC":
            save_fname = os.path.join(output_dir, f"{key_string}_auc_metrics.png")
        plt.savefig(save_fname, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to {save_fname}")

       
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_all_metrics_interactive(experiments, output_dir=".", sparsity_mode="kernel"):
    """
    Interactive Plotly version of plot_all_metrics with toggleable legend across all subplots.
    Preserves original styling: marker type, color, line style.
    """
    colors = ["#FFC20A", "#0C7BDC", "#D41159", "#004D40", "#45D86E", "#3E2717", "#FE7800"]
    markers = ['circle', 'star', 'triangle-down']
    linestyles = ['solid', 'dash', 'dot', 'dashdot']

    for key, reassignments in experiments.items():
        key_string = "_".join(key[:])

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Accuracy vs. Comm. Cost",
                "Accuracy vs. Comm. Loss",
                "Accuracy vs. Kbits",
                f"{sparsity_mode.capitalize()} Sparsity vs. Kbits"
            ]
        )

        for idx, (reassign_flag, runs) in enumerate(reassignments.items()):
            runs_sorted = sorted(runs, key=lambda x: x[0])
            pr_vals, acc_vals, cost_vals, loss_vals, kbits_vals, spar_vals = map(
                lambda vals: [to_float(v) for v in vals],
                zip(*runs_sorted)
            )

            marker_symbol = (
                markers[2] if 'prune_comm' in reassign_flag else
                markers[0] if 'fixed' in reassign_flag else
                markers[1]
            )

            line_style = linestyles[1] if 'partition_row' in reassign_flag else linestyles[0]
            color = colors[idx % len(colors)]
            hover_text = [f"pr={pr:.2f}" for pr in pr_vals]
            legendgroup = reassign_flag  # Links the four traces

            def add_trace(x, y, row, col, showlegend):
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=reassign_flag,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    line=dict(color=color, dash=line_style),
                    marker=dict(symbol=marker_symbol, size=10),
                    hovertext=hover_text,
                ), row=row, col=col)

            # Only the first trace needs `showlegend=True`
            add_trace(cost_vals, acc_vals, row=1, col=1, showlegend=True)
            add_trace(loss_vals, acc_vals, row=1, col=2, showlegend=False)
            add_trace(kbits_vals, acc_vals, row=2, col=1, showlegend=False)
            add_trace(kbits_vals, spar_vals, row=2, col=2, showlegend=False)

        fig.update_layout(
            height=800,
            width=1200,
            title_text=f"All Metrics for {key_string}",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                title="Reassign Flags"
            )
        )

        fig.update_xaxes(title_text="Comm. Cost", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)

        fig.update_xaxes(title_text="Comm. Loss", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)

        fig.update_xaxes(title_text="Kbits Transmitted", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)

        fig.update_xaxes(title_text="Kbits Transmitted", row=2, col=2)
        fig.update_yaxes(title_text=f"{sparsity_mode.capitalize()} Sparsity", row=2, col=2)

        html_path = os.path.join(output_dir, f"{key_string}_interactive.html")
        fig.write_html(html_path)
        print(f"📊 Interactive plot saved: {html_path}")


#def generate_visualizations(experiment_logs, code, dataset_root, gen_images=False, sparsity_mode="kernel", seed=1234):
def generate_visualizations(experiment_logs, test_loader, gen_images=False, sparsity_mode="kernel", device="cpu", seed=1234):
    """
    Generates accuracy vs. communication cost plots and bar charts.
    """
    # Set all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    experiments = defaultdict(lambda: defaultdict(list))
    exp_experiments = defaultdict(lambda: defaultdict(list))
    table_experiments = defaultdict(lambda: defaultdict(list))
    
    '''
    _, test_loader = dataset.get_cifar10_data(batch_size=64,
        data_folder_path = os.path.join(dataset_root, 'cifar10-data'))
    '''
    
    for folder in os.listdir(experiment_logs):
   

        folder_path = os.path.join(experiment_logs, folder)
        if not os.path.isdir(folder_path):
            continue
        
        experiment_details = parse_experiment_folder(folder)
        if not experiment_details:
            print(f"SKIPPING FOR FOLDER: {folder}, details")
            continue
        print("\n------- Running for: ", folder)
        print("\n")
        
        model_state_path = os.path.join(folder_path, "fine_tuned.pt")
        if not os.path.exists(model_state_path):
            print(f"SKIPPING FOR FOLDER: {folder}, state_path")
            continue
        
        partition_data = load_partition_data(folder_path)
        if not partition_data:
            print(f"SKIPPING FOR FOLDER: {folder}, partition_data")
            continue
        
        if experiment_details["model"] == "resnet18":
            model = models.__dict__[experiment_details['model']](nn.Conv2d, nn.BatchNorm2d, num_classes=10)
        elif experiment_details["model"] == "resnet101":
            model = models.__dict__[experiment_details['model']](get_layers('regular'), get_bn_layers('regular'), num_classes=100)
        '''
        model_r = models.__dict__[experiment_details['model']](get_layers('regular'), get_bn_layers('regular'),
                                                       num_classes=self.configs['num_classes'])
        '''

        model.load_state_dict(torch.load(model_state_path, map_location=torch.device(device)))
        
        comm_cost, total_kbits = compute_communication_cost_and_kbits(model, partition_data)
        comm_loss = compute_communication_loss(model, partition_data)
        model_spar = compute_model_sparsity(model, partition_data, mode=sparsity_mode)
        
        accuracy = load_best_accuracy(folder_path)        
        if accuracy is None:
            print(f"SKIPPING FOR FOLDER: {folder}, acc")
            continue

        latest_accuracy = load_latest_accuracy(folder_path)        
        if latest_accuracy is None:
            print(f"SKIPPING FOR FOLDER: {folder}, latest acc")
            continue

        eval_cost = load_new_comm_cost(folder_path)
        if eval_cost is None:
            print(f"SKIPPING FOR FOLDER: {folder}, eval_cost")
            continue
        eval_cost_aggregate = load_new_comm_cost2(folder_path)
        if eval_cost_aggregate is None:
            print(f"SKIPPING FOR FOLDER: {folder}, eval_cost_aggregate")
            continue
        #print("accuracy, comm_cost, eval_cost, eval_cost_aggregate: ", accuracy, comm_cost, eval_cost, eval_cost_aggregate)    
        #model_spar_kernel = compute_model_sparsity(model, partition_data, mode="kernel")

        # @@@@ AUC
        roc_auc = 0.0
        '''
        best_acc_file = os.path.join(folder_path, "best_accuracy.txt")
        if not os.path.exists(best_acc_file):
            print(f"SKIPPING FOR FOLDER: {folder}, roc_auc")
            continue

        with open(best_acc_file, 'r') as f:
            match = re.search(r'Best ROC_AUC: ([0-9\.]+)', f.read())
            if match is None:
                print("ROC_AUC Not Found in best_accuracy.txy file... Start computing it!")
                roc_auc, _ = compute_roc_auc(model, test_loader, device='cpu')
                #print(roc_auc)
                best_acc_file = os.path.join(folder_path, "best_accuracy.txt")
                with open(best_acc_file, 'a') as f:
                    f.write(f"Best ROC_AUC: {roc_auc}\n")
            else:
                roc_auc = float(match.group(1))
                print("File best_accuracy.txt contains ROC_AUC: ", roc_auc)
        '''

        key = (experiment_details['data_code'], experiment_details['model'], experiment_details['num_partition'], experiment_details['experiment_flag'])
        experiments[key]['-'.join([experiment_details['sparsity_type'], experiment_details['reassign_flag']])].append((float(experiment_details['pr_ratio']), accuracy, comm_cost, comm_loss, total_kbits, model_spar, latest_accuracy, roc_auc))
        
        exp_key = (experiment_details['data_code'], experiment_details['model'], experiment_details['num_partition'], experiment_details['topology'])
        exp_experiments[exp_key]['-'.join([experiment_details['run']])].append((float(experiment_details['pr_ratio']), accuracy, comm_cost, comm_loss, total_kbits, model_spar, latest_accuracy, roc_auc))
        

        table_experiments[exp_key]['-'.join([experiment_details['run']])].append((float(experiment_details['pr_ratio']), accuracy, comm_cost, comm_loss, total_kbits, model_spar, eval_cost, eval_cost_aggregate, latest_accuracy, roc_auc))
        
        if gen_images:
            plot_layer(model, partition_data, random.sample(range(1, len(partition_data['layers'])+1), 3), os.path.join(experiment_logs, "layer_vis"), key, '-'.join([experiment_details['sparsity_type'], experiment_details['reassign_flag']]))
    
    plot = False
    if plot:
        # Once we have 'experiments' populated, plot all metrics
        plot_all_metrics(experiments, output_dir=experiment_logs, sparsity_mode=sparsity_mode, mode="ACC")

        # Run plots per experiment
        # Valid values for mode: ACC, AUC 
        plot_all_metrics(exp_experiments, output_dir=experiment_logs, sparsity_mode=sparsity_mode, mode="ACC")
        #plot_all_metrics(exp_experiments, output_dir=experiment_logs, sparsity_mode=sparsity_mode, mode="AUC")

        # Plot best runs of prune ratios 1.0 and 0.0 as dots
        new_plot_all_metrics(exp_experiments, output_dir=experiment_logs, sparsity_mode=sparsity_mode, mode="ACC")
        #new_plot_all_metrics(exp_experiments, output_dir=experiment_logs, sparsity_mode=sparsity_mode, mode="AUC")

        #Plot in tabular form
        plot_table_metrics(table_experiments, output_dir=experiment_logs)
    
        print("✅  Visualizations saved!")
    return exp_experiments, table_experiments
    
    
if __name__ == "__main__":
    # experiment_logs_dtelecom, experiment_logs_abilene, 
    # experiment_logs_watts_strogatz, experiment_logs_barabasi_albert
    # experiment_logs_uniform
    experiment_logs_path = "experiment_logs_abilene_costs"
    data_code = "cifar100"        # valid cifar10, cifar100
    batch_size = 128
    device = "cpu"
    mode = "ACC" # valid values: ACC, AUC
    if mode == "AUC":
        device = "cuda"
        _, test_loader = get_dataset_from_code(data_code, batch_size)
    else:
        device = "cpu"
    test_loader = None
    generate_visualizations(experiment_logs_path, test_loader, gen_images=False, sparsity_mode="kernel", device=device)
