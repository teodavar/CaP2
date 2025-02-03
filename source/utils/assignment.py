import torch
import torch.nn as nn
import torch.fx as fx
import operator
import numpy as np
import pandas as pd
import time
from munkres import Munkres
from scipy.optimize import linear_sum_assignment

############################################################
#  Cost & Assignment Helpers
############################################################

def compute_cost_matrix(layer_name, layer_weights, partition):
    """
    Computes the cost of processing an output neuron/channel at each partition.

    Args:
        layer_name (str): The name of the layer.
        layer_weights (torch.Tensor): The weight tensor for the layer.
                                      - For Conv2D: (out_channels, in_channels, kernel_height, kernel_width)
                                      - For Linear: (out_features, in_features)
        partition (dict): The partition information of the model, including this information for the layer:
                          - 'num': Number of partitions
                          - 'filter_id': List of filter IDs for each partition
                          - 'channel_id': List of input channel IDs for each partition
                          - 'maps': Communication cost map between partitions

    Returns:
        np.ndarray: A cost matrix of shape (num_neurons, num_partitions), where:
                    - cost_matrix[i, j] = cost of computing output neuron `i` at partition `j`.
    """
    if layer_name not in partition:
        raise ValueError(f"Layer {layer_name} not in partition dictionary.")

    layer_part = partition[layer_name]
    num_parts = layer_part['num']
    channel_ids = layer_part['channel_id']  # input channel partition
    maps = layer_part['maps']

    w_np = layer_weights.cpu().detach().numpy()
    shape = w_np.shape
    is_conv = (len(shape) == 4)  # (out_c, in_c, kH, kW)

    out_channels = shape[0]
    cost_mat = np.zeros((out_channels, num_parts))

    # For each out_channel i, for each possible machine j
    for i in range(out_channels):
        for j in range(num_parts):
            # Sum cost if input channels come from different partition k
            for k in range(num_parts):
                if k == j:
                    continue
                input_ch_ids = channel_ids[k].astype(int)
                if is_conv:
                    # shape: (out_c, in_c, kH, kW)
                    active_w = w_np[i, input_ch_ids, :, :]
                    all_zero_per_ch = np.all(active_w == 0, axis=(1,2))
                    num_active = np.sum(~all_zero_per_ch)
                else:
                    # FC layer shape: (out_features, in_features)
                    active_w = w_np[i, input_ch_ids]
                    is_zero = (active_w == 0)
                    num_active = np.sum(~is_zero)

                if num_active > 0:
                    cost_mat[i, j] += maps[j][k] * num_active
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

def compute_cost_matrix_with_timing(layer_name, layer_weights, partition):
    """
    Adds timing to `compute_cost_matrix`.
    """

    start_time = time.time()

    if layer_name not in partition:
        raise ValueError(f"Layer {layer_name} not in partition dictionary.")

    layer_part = partition[layer_name]
    num_parts = layer_part['num']
    channel_ids = layer_part['channel_id']  # input channel partition
    maps = layer_part['maps']

    w_np = layer_weights.cpu().detach().numpy()
    shape = w_np.shape
    is_conv = (len(shape) == 4)  # (out_c, in_c, kH, kW)

    out_channels = shape[0]
    cost_mat = np.zeros((out_channels, num_parts))

    for i in range(out_channels):
        for j in range(num_parts):
            for k in range(num_parts):
                if k == j:
                    continue
                input_ch_ids = channel_ids[k].astype(int)
                if is_conv:
                    active_w = w_np[i, input_ch_ids, :, :]
                    all_zero_per_ch = np.all(active_w == 0, axis=(1, 2))
                    num_active = np.sum(~all_zero_per_ch)
                else:
                    active_w = w_np[i, input_ch_ids]
                    is_zero = (active_w == 0)
                    num_active = np.sum(~is_zero)

                if num_active > 0:
                    cost_mat[i, j] += maps[j][k] * num_active

    elapsed_time = time.time() - start_time
    print(f"[Timing] compute_cost_matrix for {layer_name}: {elapsed_time:.4f}s")
    return cost_mat

def computeassignment(cost_matrix):
    """
    Solve assignment problem (Hungarian).
    Returns list of tuples (out_channel_idx, machine_idx, cost).
    """
    n_tasks, n_machines = cost_matrix.shape
    if n_tasks <= n_machines:
        c_expanded = cost_matrix
        cap = 1
    else:
        # replicate machines if needed
        cap = int(np.ceil(n_tasks / n_machines))
        c_expanded = np.repeat(cost_matrix, cap, axis=1)

    m = Munkres()
    indexes = m.compute(c_expanded)

    results = []
    for row, col in indexes:
        if row < n_tasks:
            real_machine = col // cap
            cost_val = cost_matrix[row, real_machine]
            results.append((row, real_machine, cost_val))
    return results

def computeassignment_scipy(cost_matrix):
    """
    Equivalent to `computeassignment`, but uses SciPy's 
    linear_sum_assignment for faster performance in most cases.

    Args:
      cost_matrix (np.ndarray): shape=(n_tasks, n_machines)

    Returns:
      A list of tuples (row, machine, cost_val) giving the assignment:
        - row = index of out_channel
        - machine = partition index
        - cost_val = cost_matrix[row, machine]
    """

    n_tasks, n_machines = cost_matrix.shape
    if n_tasks <= n_machines:
        c_expanded = cost_matrix
        cap = 1
    else:
        cap = int(np.ceil(n_tasks / n_machines))
        c_expanded = np.repeat(cost_matrix, cap, axis=1)

    # Use SciPy's linear_sum_assignment on the expanded matrix
    row_ind, col_ind = linear_sum_assignment(c_expanded)

    results = []
    for row, col in zip(row_ind, col_ind):
        if row < n_tasks:
            real_machine = col // cap
            cost_val = cost_matrix[row, real_machine]
            results.append((row, real_machine, cost_val))

    return results


def computeassignment_with_timing(cost_matrix, save_matrix=False):
    """
    Adds timing to `computeassignment`.
    """

    start_time = time.time()

    n_tasks, n_machines = cost_matrix.shape
    if n_tasks <= n_machines:
        c_expanded = cost_matrix
        cap = 1
    else:
        cap = int(np.ceil(n_tasks / n_machines))
        c_expanded = np.repeat(cost_matrix, cap, axis=1)
    
    print("Cost matrix shape:", cost_matrix.shape)
    print("Expanded matrix shape:", c_expanded.shape)
    print("cap =", cap)

    m = Munkres()
    indexes = m.compute(c_expanded)

    results = []
    for row, col in indexes:
        if row < n_tasks:
            real_machine = col // cap
            cost_val = cost_matrix[row, real_machine]
            results.append((row, real_machine, cost_val))

    elapsed_time = time.time() - start_time
    print(f"[Timing] computeassignment: {elapsed_time:.4f}s")
    
    if save_matrix and elapsed_time > 20:
        filename = f"cost_matrix_{int(time.time())}.csv"
        pd.DataFrame(cost_matrix).to_csv(filename, index=False, header=False)
        print(f"Execution time exceeded 20s. Cost matrix saved as {filename}.")
        
    if c_expanded.shape[0] == 512:
        matrix_stats = analyze_cost_matrix_sparsity(cost_matrix)
        print("Cost matrix stats:", matrix_stats)
    return results

def computeassignment_with_timing_scipy(cost_matrix, save_matrix=False):
    """
    Equivalent to `computeassignment_with_timing`, but uses SciPy's 
    linear_sum_assignment for faster performance in most cases.

    Args:
      cost_matrix (np.ndarray): shape=(n_tasks, n_machines)
      save_matrix (bool): whether to save the cost matrix if it takes >20s

    Returns:
      A list of tuples (row, machine, cost_val) giving the assignment:
        - row = index of out_channel
        - machine = partition index
        - cost_val = cost_matrix[row, machine]
    """
    start_time = time.time()

    n_tasks, n_machines = cost_matrix.shape
    if n_tasks <= n_machines:
        c_expanded = cost_matrix
        cap = 1
    else:
        cap = int(np.ceil(n_tasks / n_machines))
        c_expanded = np.repeat(cost_matrix, cap, axis=1)

    print("Cost matrix shape:", cost_matrix.shape)
    print("Expanded matrix shape:", c_expanded.shape)
    print("cap =", cap)

    # Use SciPy's linear_sum_assignment on the expanded matrix
    row_ind, col_ind = linear_sum_assignment(c_expanded)

    results = []
    for row, col in zip(row_ind, col_ind):
        if row < n_tasks:
            real_machine = col // cap
            cost_val = cost_matrix[row, real_machine]
            results.append((row, real_machine, cost_val))

    elapsed_time = time.time() - start_time
    print(f"[Timing] computeassignment (SciPy): {elapsed_time:.4f}s")

    # Optionally save the matrix if it took > 20s
    if save_matrix and elapsed_time > 20:
        filename = f"cost_matrix_{int(time.time())}.csv"
        pd.DataFrame(cost_matrix).to_csv(filename, index=False, header=False)
        print(f"Execution time exceeded 20s. Cost matrix saved as {filename}.")

    # Optionally analyze sparsity if matrix is large (e.g. 512x4)
    if c_expanded.shape[0] == 512:
        matrix_stats = analyze_cost_matrix_sparsity(cost_matrix)
        print("Cost matrix stats:", matrix_stats)

    return results


def list_to_partition(assignments, num_parts):
    """
    Convert assignment (per out_channel -> machine) to a new partition structure
    """
    assignments = np.array(assignments)

    new_part = []
    for j in range(num_parts):
        new_part.append(np.where(assignments == j)[0])
    return new_part


############################################################
#  Residual (Add) Helpers
############################################################

def compute_combined_cost_matrix(layerA_name, wA,
                                 layerB_name, wB,
                                 partition):
    """
    cost[i,j] = costA[i,j] + costB[i,j]
    We unify the assignment for out_channel i in both layers A and B.
    """
    costA = compute_cost_matrix(layerA_name, wA, partition)
    costB = compute_cost_matrix(layerB_name, wB, partition)

    if costA.shape != costB.shape:
        raise ValueError(f"Shapes mismatch: {costA.shape} vs {costB.shape} for {layerA_name} & {layerB_name}")

    return costA + costB

def unify_assignments_for_add(layerA, layerB,
                              model, partition_dict, node_map, gm,
                              relationship="none"):
    """
    1) Retrieve each layer's original partition & parent partition
    2) Build combined cost matrix
    3) Solve assignment
    4) Update partition for either:
       (a) both convs (standard case), or
       (b) only the child conv if one is parent of the other.
    """

    parentA = get_parent_partition(layerA, partition_dict, node_map, gm)
    if parentA is not None:
        partition_dict[layerA]['channel_id'] = parentA['filter_id'] 

    parentB = get_parent_partition(layerB, partition_dict, node_map, gm)
    if parentB is not None:
        partition_dict[layerB]['channel_id'] = parentB['filter_id'] 

    wA = model.state_dict()[layerA]
    wB = model.state_dict()[layerB]

    combined_cost = compute_combined_cost_matrix(layerA, wA, layerB, wB, partition_dict)
    assignment_sol = computeassignment_scipy(combined_cost)
    assignment_sol.sort(key=lambda x: x[0])
    assigns = [m for (ch,m,c) in assignment_sol]

    newA = list_to_partition(assigns, partition_dict[layerA]['num'])
    newB = list_to_partition(assigns, partition_dict[layerB]['num'])

    if relationship == "A_is_parent":
        # Only update B (child)
        partition_dict[layerB]['filter_id'] = np.asarray(newB)

    elif relationship == "B_is_parent":
        # Only update A (child)
        partition_dict[layerA]['filter_id'] = np.asarray(newA)

    else:
        # No parent-child relationship; update both
        partition_dict[layerA]['filter_id'] = np.asarray(newA)
        partition_dict[layerB]['filter_id'] = np.asarray(newB)
        
def unify_assignments_for_add_with_timing(layerA, layerB, model, partition_dict, node_map, gm, relationship="none"):
    """
    Timing-enhanced version of `unify_assignments_for_add` to log timing for each step.
    """

    timing_info = {}

    # Retrieve parent partitions
    start_time = time.time()
    parentA = get_parent_partition(layerA, partition_dict, node_map, gm)
    timing_info['get_parent_partition_A'] = time.time() - start_time

    if parentA is not None:
        partition_dict[layerA]['channel_id'] = parentA['filter_id']

    start_time = time.time()
    parentB = get_parent_partition(layerB, partition_dict, node_map, gm)
    timing_info['get_parent_partition_B'] = time.time() - start_time

    if parentB is not None:
        partition_dict[layerB]['channel_id'] = parentB['filter_id']

    # Compute the combined cost matrix
    start_time = time.time()
    wA = model.state_dict()[layerA]
    wB = model.state_dict()[layerB]
    combined_cost = compute_combined_cost_matrix(layerA, wA, layerB, wB, partition_dict)
    timing_info['compute_combined_cost_matrix'] = time.time() - start_time

    # Solve the assignment problem
    start_time = time.time()
    assignment_sol = computeassignment_with_timing(combined_cost)
    timing_info['compute_assignment'] = time.time() - start_time

    # Convert assignments to new partitions
    start_time = time.time()
    assignment_sol.sort(key=lambda x: x[0])
    assigns = [m for (ch, m, c) in assignment_sol]
    newA = list_to_partition(assigns, partition_dict[layerA]['num'])
    newB = list_to_partition(assigns, partition_dict[layerB]['num'])
    timing_info['list_to_partition'] = time.time() - start_time

    # Update the partition dictionary
    start_time = time.time()
    if relationship == "A_is_parent":
        partition_dict[layerB]['filter_id'] = np.asarray(newB)
    elif relationship == "B_is_parent":
        partition_dict[layerA]['filter_id'] = np.asarray(newA)
    else:
        partition_dict[layerA]['filter_id'] = np.asarray(newA)
        partition_dict[layerB]['filter_id'] = np.asarray(newB)
    timing_info['update_partition_dict'] = time.time() - start_time

    # Print timing results for this function call
    total_time = sum(timing_info.values())
    print(f"Timing for unify_assignments_for_add ({layerA}, {layerB}):")
    for step, elapsed in timing_info.items():
        print(f"  {step}: {elapsed:.4f}s")
    print(f"  Total: {total_time:.4f}s")


############################################################
#  Graph Analysis: Pre-Scan for Add Pairs
############################################################

def backtrack_to_convs(start_node, gm):
    """
    From 'start_node', keep going backwards in the graph until we find all
    call_module nodes that are nn.Conv2d (and thus appear in our partition).
    This accounts for BN/ReLU etc. in between.
    """
    visited = set()
    stack = [start_node]
    conv_nodes = []
    named_mods = dict(gm.named_modules())

    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)

        if cur.op == "call_module":
            submod = named_mods[cur.target]
            if isinstance(submod, nn.Conv2d):
                conv_nodes.append(cur)
                # do not traverse further from a found conv
                continue

        # else keep going upwards
        for arg in cur.args:
            if isinstance(arg, fx.Node):
                stack.append(arg)

    return conv_nodes


def is_ancestor(nodeA, nodeB, gm):
    """
    Return True if nodeA is a direct or indirect ancestor of nodeB in the FX graph.
    We do a forward traversal from nodeA, checking all nodeA.users, then their users, etc.
    """
    visited = set()
    queue = [nodeA]
    while queue:
        current = queue.pop(0)
        if current == nodeB:
            return True
        for user in current.users:
            if user not in visited:
                visited.add(user)
                queue.append(user)
    return False


def build_add_pairs(gm, partition_dict, node_map):
    """
    We do a pre-scan of the graph to find any add nodes and identify exactly
    which two conv nodes feed them. We'll store them in a map:
       add_node -> (layerA, layerB, relationship)

    where relationship in {"none", "A_is_parent", "B_is_parent"}.
    """
    add_pairs = {}  # add_node -> (layerA, layerB, relationship)
    
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in (operator.add, torch.add):
            # look for 2-arg add
            if len(node.args) != 2:
                continue
            lhs, rhs = node.args
            if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
                continue

            # backtrack each side to find conv(s)
            lhs_convs = backtrack_to_convs(lhs, gm)
            rhs_convs = backtrack_to_convs(rhs, gm)

            # If exactly 1 conv on each side, check if they are in partition_dict
            if len(lhs_convs) == 1 and len(rhs_convs) == 1:
                convA_node = lhs_convs[0]
                convB_node = rhs_convs[0]
                layerA = node_map.get(convA_node, None)
                layerB = node_map.get(convB_node, None)

                # must be in partition_dict
                if layerA and layerB and layerA in partition_dict and layerB in partition_dict:
                    # Check relationship
                    if is_ancestor(convA_node, convB_node, gm):
                        relation = "A_is_parent"
                    elif is_ancestor(convB_node, convA_node, gm):
                        relation = "B_is_parent"
                    else:
                        relation = "none"

                    add_pairs[node] = (layerA, layerB, relation)

    return add_pairs


def build_node_map(gm):
    """
    Return a dict: node -> 'submodule_name.weight'
    if node is a call_module that might correspond to partition_dict keys.
    """
    nmap = {}
    for n in gm.graph.nodes:
        if n.op == "call_module":
            submod_name = n.target
            candidate_key = submod_name + ".weight"
            nmap[n] = candidate_key
    return nmap


############################################################
#  Parent Partition Logic
############################################################

def get_parent_partition(layer_name, partition_dict, node_map, gm):
    """
    Finds the immediate parent partition of `layer_name` by analyzing the FX graph.
    We backtrack to the nearest conv node and use partition_dict[conv_layer_name].
    
    Returns a dict: { 'num': <int>, 'maps': <2D array/list>, 'filter_id': [arrays per partition] }
    or None if no parent partition found.
    
    Args:
      layer_name (str): e.g. "conv2.weight"
      partition_dict (dict): your layer partition dictionary
      node_map (dict): maps fx.Node -> "layer_name.weight" from build_node_map(gm)
      gm (fx.GraphModule): the symbolic-traced graph
    """
    # Build a reverse map so we can go from layer_name -> fx.Node
    reverse_map = {v: k for k, v in node_map.items()}
    if layer_name not in reverse_map:
        # No fx Node for this layer => no parent
        return None

    this_node = reverse_map[layer_name]

    assert isinstance(this_node, torch.fx.Node)

    # If the current node has no inputs, or its inputs are not FX Nodes, there's no parent partition
    if not this_node.args:
        return None

    # We'll pick the first Node argument we find (in many cases it's just one input for a Conv)
    parent_candidate = None
    for arg in this_node.args:
        if isinstance(arg, torch.fx.Node):
            parent_candidate = arg
            break
    if parent_candidate is None:
        return None

    # Now we "backtrack to convs" from that input node,
    # skipping BN/ReLU/etc. to see if there's exactly one upstream Conv.
    conv_nodes = backtrack_to_convs(parent_candidate, gm)
    if not conv_nodes:
        # Possibly the input is the raw input x or no real conv found
        return None

    # If there's exactly 1 conv ancestor, use it; if more, pick the first
    conv_parent_node = conv_nodes[0]
    if conv_parent_node in node_map:
        conv_parent_name = node_map[conv_parent_node]  # e.g. "conv1.weight"
        if conv_parent_name in partition_dict:
            pinfo = partition_dict[conv_parent_name]
            return {
                'num': pinfo['num'],
                'maps': pinfo['maps'],
                'filter_id': pinfo['filter_id']
            }

    return None


############################################################
#  MAIN update_assignments
############################################################

def update_assignments(model, configs, model_graph):
    """
    We:
      Do a topological pass:
         - If node is a conv in partition_dict and not in add_pairs => single-layer assignment
         - If node is a conv that is in add_pairs => skip (we unify at the add node)
         - If node is an add => unify those conv parents
         - Otherwise => do nothing

    Assumption: The input itself will not be used as a residual connection.
    """
    # Declare structures
    partition_dict = configs['partition']
    gm = model_graph['graph']
    node_map = model_graph['node_map']
    named_mods = model_graph['named_mods']
    add_node_pairs_map = model_graph['addition_nodes']
    conv_in_add = model_graph['addition_set']
    parents_in_add = model_graph['parents']
    
    # Topological pass
    for node in gm.graph.nodes:
        if node.op == "call_module":
            submod_name = node.target
            submod = named_mods[submod_name]
            layer_key = submod_name + ".weight"

            # Check if it's a conv in partition_dict
            if (layer_key in partition_dict) and isinstance(submod, nn.Conv2d):
                # If in add but not a parent => skip (will unify in add)
                if layer_key in conv_in_add:
                    #print(f"[Skip for Add] {layer_key}")
                    continue

                # Not in any add or also a parent => single-layer assignment
                else:
                    #print(f"[Single] Assign for {layer_key}")
                    parent_part = get_parent_partition(layer_key, partition_dict, node_map, gm)
                    if parent_part is not None:
                        partition_dict[layer_key]['channel_id'] = parent_part['filter_id']
                    w = model.state_dict()[layer_key]
                    cost_mat = compute_cost_matrix(layer_key, w, partition_dict)
                    sol = computeassignment_scipy(cost_mat)
                    sol.sort(key=lambda x: x[0])
                    assigns = [m for (ch, m, c) in sol]
                    newp = list_to_partition(assigns, partition_dict[layer_key]['num'])
                    partition_dict[layer_key]['filter_id'] = np.asarray(newp)

        elif node in add_node_pairs_map:
            # This is an add node
            (layerA, layerB, relationship) = add_node_pairs_map[node]
            #print(f"[Residual Add] {node.name} => {layerA} + {layerB}; relationship={relationship}")
            unify_assignments_for_add(layerA, layerB, model, partition_dict, node_map, gm,
                                      relationship=relationship)

    print("Finished update_assignments.")

def update_assignments_with_timing(model, configs, model_graph):
    """
    Benchmark-enhanced version of `update_assignments` to log timing information for each step,
    using the timing-instrumented helper functions.
    """

    # Declare structures
    partition_dict = configs['partition']
    gm = model_graph['graph']
    node_map = model_graph['node_map']
    named_mods = model_graph['named_mods']
    add_node_pairs_map = model_graph['addition_nodes']
    conv_in_add = model_graph['addition_set']
    parents_in_add = model_graph['parents']

    total_time = 0
    per_layer_timing = {}

    # Topological pass
    for node in gm.graph.nodes:
        start_time = time.time()

        if node.op == "call_module":
            submod_name = node.target
            submod = named_mods[submod_name]
            layer_key = submod_name + ".weight"

            # Check if it's a conv in partition_dict
            if (layer_key in partition_dict) and isinstance(submod, nn.Conv2d):
                # If in add but not a parent => skip (will unify in add)
                if layer_key in conv_in_add:
                    continue

                # Not in any add or also a parent => single-layer assignment
                parent_part = get_parent_partition(layer_key, partition_dict, node_map, gm)
                if parent_part is not None:
                    partition_dict[layer_key]['channel_id'] = parent_part['filter_id']

                w = model.state_dict()[layer_key]

                # Use timing-enabled helpers
                cost_start = time.time()
                cost_mat = compute_cost_matrix(layer_key, w, partition_dict)
                cost_time = time.time() - cost_start

                assign_start = time.time()
                sol = computeassignment_with_timing(cost_mat)
                assign_time = time.time() - assign_start

                sol.sort(key=lambda x: x[0])
                assigns = [m for (ch, m, c) in sol]
                newp = list_to_partition(assigns, partition_dict[layer_key]['num'])
                partition_dict[layer_key]['filter_id'] = np.asarray(newp)

                per_layer_timing[layer_key] = {
                    'cost_matrix': cost_time,
                    'assignment': assign_time,
                    'total': time.time() - start_time
                }

        elif node in add_node_pairs_map:
            # This is an add node
            (layerA, layerB, relationship) = add_node_pairs_map[node]

            unify_start = time.time()
            unify_assignments_for_add_with_timing(
                layerA, layerB, model, partition_dict, node_map, gm, relationship=relationship
            )
            unify_time = time.time() - unify_start

            per_layer_timing[f"add_node_{node.name}"] = {'unify_assignments': unify_time}

        total_time += time.time() - start_time

    # Print timing results
    print("Timing Results:")
    for layer, timings in per_layer_timing.items():
        print(f"{layer}: {timings}")
    print(f"Total update_assignments time: {total_time:.4f}s")

############################################################
#  Example or Test
############################################################
if __name__ == "__main__":

    class SimpleRes(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.shortcut = nn.Conv2d(64, 64, 1)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        def forward(self, x):
            y = self.conv1(x)
            y = self.bn1(y)
            y = self.relu(y)
            z = self.conv2(y)
            sc = self.shortcut(y)
            out = z + sc   # add op
            out = self.conv3(out)
            return out

    model = SimpleRes()

    # Example partition_dict
    partition_dict = {
        'conv1.weight': {
            'num': 2,
            'filter_id': [np.array([]), np.array([])],
            'channel_id': [np.array([0,1,2]), np.array([])],  # input is 3 channels in partition 0
            'maps': [[0,1],[1,0]]
        },
        'conv2.weight': {
            'num': 2,
            'filter_id': [np.array([]), np.array([])],
            'channel_id': [np.array([]), np.array([])],
            'maps': [[0,1],[1,0]]
        },
        'shortcut.weight': {
            'num': 2,
            'filter_id': [np.array([]), np.array([])],
            'channel_id': [np.array([]), np.array([])],
            'maps': [[0,1],[1,0]]
        },
        'conv3.weight': {
            'num': 2,
            'filter_id': [np.array([]), np.array([])],
            'channel_id': [np.array([]), np.array([])],
            'maps': [[0,1],[1,0]]
        }
    }

    configs = {'partition': partition_dict}

    update_assignments(model, configs)

    # Inspect final partitions
    for k, v in configs['partition'].items():
        print("\nLayer:", k)
        print("filter_id:", [arr.tolist() for arr in v['filter_id']])
        print("channel_id:", [arr.tolist() for arr in v['channel_id']])
        print("maps:", v['maps'])
