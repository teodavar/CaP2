import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import wandb
from munkres import Munkres
from scipy.optimize import linear_sum_assignment

############################################################
#  Cost & Assignment Helpers
############################################################

def compute_cost_matrix(layer_name, layer_weights, partition, use_comm_cost=False):
    """
    Computes the cost of processing an output neuron/channel at each partition.

    Args:
        layer_name (str): The name of the layer.
        layer_weights (torch.Tensor): The weight tensor for the layer.
                                      - For Conv2D: (out_channels, in_channels, kernel_height, kernel_width)
                                      - For Linear: (out_features, in_features)
        partition (dict): The partition information of the model, including this information for the layer

    Returns:
        np.ndarray: A cost matrix of shape (num_neurons, num_partitions), where:
                    - cost_matrix[i, j] = cost of computing output neuron `i` at partition `j`.
    """
    
    start_time = time.time()
    
    if layer_name not in partition:
        raise ValueError(f"Layer {layer_name} not in partition dictionary.")

    layer_part = partition[layer_name]
    num_parts = partition['num']
    maps = partition['maps']
    parents = layer_part.get('parents', [])  # List of parent layer names

    w_np = layer_weights.cpu().detach().numpy()
    is_conv = (len(w_np.shape) == 4)  # (out_c, in_c, kH, kW)

    out_channels = w_np.shape[0]
    cost_mat = np.zeros((out_channels, num_parts))
    
    # Retrieve any 'outsize' scaling factor for this layer (default to 1 if not present)
    outsize = layer_part.get('outsize', 1.0)

    # For each out_channel i, for each possible machine j
    for i in range(out_channels):
        for j in range(num_parts):
            cost_ij = 0.0
            
            # Sum cost contributions from each parent layer
            for parent_layer in parents:
                if parent_layer not in partition:
                    raise ValueError(f"Parent layer {parent_layer} not found in partition dictionary.")

                # Get parent partitioning
                parent_filter_ids = partition[parent_layer]['filter_id']

                # For each input-partition k
                for k in range(num_parts):
                    if k == j:
                        continue
                    # The input channels belonging to machine k for this parent layer
                    input_ch_ids = np.asarray(parent_filter_ids[k], dtype=int)

                    if input_ch_ids.size == 0:
                        continue

                    # Gather the weights from these input channels
                    if is_conv:
                        # shape: (out_c, in_c, kH, kW)
                        w_sub = w_np[i, input_ch_ids, :, :]
                    else:
                        # shape: (out_features, in_features)
                        w_sub = w_np[i, input_ch_ids]
                    if not use_comm_cost:
                        # Sum the absolute values of these weights
                        sum_abs = np.sum(np.abs(w_sub))

                        # If there's any magnitude, add communication cost
                        # scaled by the sum of those weights
                        if sum_abs > 0:
                            cost_ij += maps[k][j] * sum_abs * outsize
                    else:
                        shape = w_sub.shape
                        if np.any(w_sub.reshape(shape[0], shape[1], -1)):
                            cost_ij += maps[k][j] * outsize
                        
            cost_mat[i, j] = cost_ij
    
    elapsed_time = time.time() - start_time
    #print(f"[Timing] compute_cost_matrix for {layer_name}: {elapsed_time:.4f}s")
    return cost_mat

def analyze_cost_matrix_sparsity(cost_matrix, near_zero_threshold=1e-9):
    """
    Analyzes the sparsity and distribution of a cost matrix.
    
    Args:
        cost_matrix (np.ndarray): The cost matrix (2D) to analyze.
        near_zero_threshold (float): Values with absolute value below this 
                                     threshold will be counted as 'near zero'.
    
    Returns:
        A dict of statistics about the matrix:
            {
                'shape': (rows, cols),
                'num_elements': int,
                'num_zeros': int,
                'frac_zeros': float,
                'num_near_zero': int,
                'frac_near_zero': float,
                'min_value': float,
                'max_value': float,
                'mean_value': float,
                'std_value': float
            }
    """
    mat = cost_matrix
    stats = {}
    
    stats['shape'] = mat.shape
    stats['num_elements'] = mat.size
    
    # Count exact zeros
    zero_mask = (mat == 0)
    stats['num_zeros'] = np.count_nonzero(zero_mask)
    stats['frac_zeros'] = stats['num_zeros'] / mat.size
    
    # Count near-zero values
    near_zero_mask = np.abs(mat) < near_zero_threshold
    stats['num_near_zero'] = np.count_nonzero(near_zero_mask)
    stats['frac_near_zero'] = stats['num_near_zero'] / mat.size
    
    # Basic distribution stats
    stats['min_value'] = np.min(mat)
    stats['max_value'] = np.max(mat)
    stats['mean_value'] = np.mean(mat)
    stats['std_value'] = np.std(mat)
    
    return stats

def add_virtualmachines(cost_matrix, budget=None):
    """
    Expands the cost matrix based on machine capacities.

    Args:
      cost_matrix (np.ndarray): shape=(n_tasks, n_machines)
      budget (list or np.ndarray, optional): Capacity allocation per machine. 
          If None, assumes uniform allocation.

    Returns:
      tuple: 
        - cost_expanded (np.ndarray): Expanded cost matrix with virtual machines.
        - machine_capacities (list): Number of virtual copies for each machine.
    """
    n_tasks, n_machines = cost_matrix.shape

    if budget is None:
        # Default to equal distribution
        capacity_per_machine = int(np.ceil(n_tasks / n_machines))
        machine_capacities = [capacity_per_machine] * n_machines
    else:
        # Ensure the budget sums to 1 (100%)
        assert np.isclose(sum(budget), 1.0), f"Budget percentages must sum to 1. Got {sum(budget)} instead."

        # Compute number of virtual machines per machine based on percentage
        raw_capacities = np.array(budget) * n_tasks
        machine_capacities = np.round(raw_capacities).astype(int)

        # Adjust last machine to ensure total tasks match
        difference = n_tasks - sum(machine_capacities)
        machine_capacities[-1] += difference  # Fix rounding issues

    # Expand cost matrix by repeating each column according to its assigned capacity
    cost_expanded = np.repeat(cost_matrix, machine_capacities, axis=1)

    return cost_expanded, machine_capacities

def computeassignment_scipy(cost_matrix, budget=None):
    """
    Uses SciPy's linear_sum_assignment for faster performance in most cases.
    This version supports custom budget allocation for machines.

    Args:
      cost_matrix (np.ndarray): shape=(n_tasks, n_machines)
      budget (list or np.ndarray, optional): Capacity allocation per machine. 
          If None, assumes uniform allocation.

    Returns:
      A list of tuples (row, machine, cost_val) giving the assignment:
        - row = index of out_channel
        - machine = partition index
        - cost_val = cost_matrix[row, machine]
    """
    start_time = time.time()
    
    # Expand the cost matrix based on budget
    cost_expanded, machine_capacities = add_virtualmachines(cost_matrix, budget)
    n_tasks, _ = cost_matrix.shape

    # Use SciPy's linear_sum_assignment on the expanded matrix
    row_ind, col_ind = linear_sum_assignment(cost_expanded)

    # Map assignments back to real machines
    results = []
    machine_mapping = []
    
    # Create an array that maps virtual machines back to original ones
    for machine_idx, cap in enumerate(machine_capacities):
        machine_mapping.extend([machine_idx] * cap)  # Repeat each machine based on capacity
    
    total_cost = 0
    for row, col in zip(row_ind, col_ind):
        if row < n_tasks:  # Ensure we're within real task indices
            real_machine = machine_mapping[col]
            cost_val = cost_matrix[row, real_machine]
            total_cost += cost_val
            results.append((row, real_machine, cost_val))

    
    elapsed_time = time.time() - start_time
    #print(f"[Timing] computeassignment: {elapsed_time:.4f}s")
    return results, total_cost

def list_to_partition(assignments, num_parts):
    """
    Convert assignment (out_channel -> machine) to partition structure.
    """
    assignments = np.array(assignments)
    return [np.where(assignments == j)[0] for j in range(num_parts)]

############################################################
#  MAIN update_assignments
############################################################

def update_assignments(model, configs, use_wandb=False, batch_number=None):
    """
    We:
      Do a topological pass and assign nodes

    Assumption: The input itself will not be used as a residual connection.
    """
    
    start_time = time.time()
    
    # Declare structures
    partition_dict = configs['partition']
    model_graph = partition_dict['model_graph']
    gm = model_graph['graph']
    node_map = model_graph['node_map']
    add_node_pairs_map = model_graph['addition_nodes']
    named_mods = dict(gm.named_modules())
    
    # Topological pass
    for node in gm.graph.nodes:
        if node.op == "call_module":
            submod_name = node.target
            submod = named_mods[submod_name]
            layer_key = submod_name + ".weight"

            # Check if it's a conv/linear in partition_dict
            if (layer_key in partition_dict) and isinstance(submod, (nn.Conv2d, nn.Linear)):
                    w = model.state_dict()[layer_key]
                    cost_mat = compute_cost_matrix(layer_key, w, partition_dict)
                    assert len(configs['partition'][layer_key]['budget']) > 0, "Error: Budget list is empty!"
                    sol, total_cost = computeassignment_scipy(cost_mat, configs['partition'][layer_key]['budget'])
                    sol.sort(key=lambda x: x[0])
                    assigns = [m for (ch, m, c) in sol]
                    newp = list_to_partition(assigns, partition_dict['num'])
                    partition_dict[layer_key]['filter_id'] = np.asarray(newp)
    return total_cost

############################################################
#  Example or Test
############################################################
if __name__ == "__main__":
    class SimpleRes(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.shortcut = nn.Conv2d(64, 64, 1)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        def forward(self, x):
            y = self.conv1(x)
            z = self.conv2(y)
            sc = self.shortcut(y)
            out = z + sc
            return self.conv3(out)

    model = SimpleRes()
    partition_dict = {
        'conv1.weight': {'num': 2, 'filter_id': [np.array([]), np.array([])], 'parents': [], 'maps': [[0,1],[1,0]]},
        'conv2.weight': {'num': 2, 'filter_id': [np.array([]), np.array([])], 'parents': ['conv1.weight'], 'maps': [[0,1],[1,0]]}
    }
    configs = {'partition': partition_dict, 'model_graph': {'graph': fx.symbolic_trace(model), 'named_mods': dict(model.named_modules())}}

    update_assignments(model, configs)
        
