import os
import yaml
import torch
import numpy as np
import torch.fx as fx
import torch.nn as nn
import operator
import random
import time
import itertools
import re
import copy

def create_partition_legacy(configs, model):

    class MyDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(MyDumper, self).increase_indent(flow=flow, indentless=indentless)

    def represent_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    MyDumper.add_representer(list, represent_list)

    # partition = {}
    num_partition = {}
    if isinstance(configs['num_partition'], int):
        # Todo: automatically set bn_partition
        ratio_partition, map_partition = {}, {}
        bn_partition = [int(configs['num_partition'])] * 9
        num = int(configs['num_partition'])
        maps = np.ones((num,num))
        for name, W in itertools.chain(model.named_parameters() , list({'inputs':None}.items())):
            if name=='inputs' or (len(W.size()) == 4) or (len(W.size()) == 2):
                if not 'out' in name:                  
                    num_partition[name] = num
                    ratio_partition[name] = [1]*num
                    np.fill_diagonal(maps, 0)
                    map_partition[name] = maps.astype(int).tolist()
        full_dict = {'bn_partitions': bn_partition, 'partitions': ratio_partition, 'maps': map_partition}
        with open(configs['partition_path'], "w") as stream:
            yaml.dump(full_dict, stream, Dumper=MyDumper, default_flow_style=False)
    else:
        raise Exception("num_partition must be an integer to initialize partition")
        
def create_partition(configs, model):

    class MyDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(MyDumper, self).increase_indent(flow=flow, indentless=indentless)

    def represent_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    MyDumper.add_representer(list, represent_list)

    num_partition = {}
    if isinstance(configs['num_partition'], int):
        bn_partition = [int(configs['num_partition'])] * 9
        num = int(configs['num_partition'])
        maps = np.ones((num, num))
        np.fill_diagonal(maps, 0)
        
        budget = configs.get('budget', [1]*num)
        input_part = configs.get('input_partition', [1]*num)
        layers = ['inputs']
        budgets = []
        
        for name, W in itertools.chain(model.named_parameters()):
            if (len(W.size()) == 4 or len(W.size()) == 2) and 'out' not in name:
                layers.append(name)
                budgets.append(list(budget))
        
        full_dict = {
            'bn_partitions': bn_partition,
            'maps': maps.astype(int).tolist(),
            'num': configs['num_partition'],
            'initial_budget': budget,
            'budgets': budgets,
            'input_partition': input_part,
            'layers': layers
        }
        
        with open(configs['partition_path'], "w") as stream:
            yaml.dump(full_dict, stream, Dumper=MyDumper, default_flow_style=False)
    else:
        raise Exception("num_partition must be an integer to initialize partition")

def save_partition(configs, epoch=0, save_path=None):
    class MyDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(MyDumper, self).increase_indent(flow=flow, indentless=indentless)

    def represent_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    
    MyDumper.add_representer(list, represent_list)
    
    partition_copy = copy.deepcopy({k: v for k, v in configs['partition'].items() if k != 'model_graph'})
    
    # Convert filter_id lists of lists into standard lists
    for layer in partition_copy:
        if isinstance(partition_copy[layer], dict) and 'filter_id' in partition_copy[layer]:
            partition_copy[layer]['filter_id'] = [sublist.tolist() for sublist in partition_copy[layer]['filter_id'] if isinstance(sublist, np.ndarray)]
        if isinstance(partition_copy[layer], dict) and 'budget' in partition_copy[layer]:
            partition_copy[layer]['budget'] = [sublist.tolist() for sublist in partition_copy[layer]['budget'] if isinstance(sublist, np.ndarray)]
        if isinstance(partition_copy[layer], np.ndarray):
            partition_copy[layer] = partition_copy[layer].tolist()
            
    
    if save_path is None:
        save_path = re.sub(r'\.ya?ml$', '', configs['partition_path'])
        save_path = save_path + f'_{epoch}.yaml'
    
    with open(save_path, "w") as stream:
        yaml.dump(partition_copy, stream, Dumper=MyDumper, default_flow_style=False)
    
    print(f"Partition saved to {save_path}")
    
def load_partition(configs, model):
    """
    Loads the partition YAML file and reconstructs the partition dictionary.
    Converts any list-based `filter_id` back into NumPy arrays.
    """
    with open(configs['partition_path'], "r") as stream:
        partition_data = yaml.safe_load(stream)

    # Convert lists back to NumPy arrays where applicable
    for key, value in partition_data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if (sub_key == 'filter_id' or sub_key == 'budget') and isinstance(sub_value, list):  # Convert lists back to NumPy arrays
                    partition_data[key][sub_key] = [np.array(li) for li in sub_value]
    
    configs['partition'] = partition_data
    
    model_graph = fx.symbolic_trace(model)
    node_map = build_node_map(model_graph)
    add_node_pairs_map = build_add_pairs(model_graph, partition_data, node_map)
    
    configs['partition']['model_graph'] = {
        'graph': model_graph,
        'node_map': node_map,
        'addition_nodes': add_node_pairs_map,
    }
    
    print("Partition loaded and model graph reconstructed.")
    return configs

def generate_partition(configs, model):
    with open(configs['partition_path'], "r") as stream:
        raw_dict = yaml.safe_load(stream)
    
    bn_partition = raw_dict['bn_partitions']
    num_partitions = raw_dict['num']
    i_budget = raw_dict['initial_budget']
    budgets = raw_dict['budgets']
    budgets = [np.array(b, dtype=np.float64)/sum(b) for b in budgets]
    layers = raw_dict['layers']
    maps = raw_dict['maps']
    input_part = np.array(raw_dict['input_partition'])
    input_part /= input_part.sum()
    
    if len(budgets[0]) != num_partitions:
        raise ValueError(f"Budget length {len(budgets[0])} does not match num_partitions {num_partitions}.")
    if len(input_part) != num_partitions:
        raise ValueError(f"Input partitioning length {len(input_part)} does not match num_partitions {num_partitions}.")
     
    partition = {
        'bn_partition': bn_partition,
        'num': num_partitions,
        'maps': maps,
        'initial_budget': budgets,
        'input_partition': input_part.tolist(),
        'layers': layers,
    }
    
    model_graph = fx.symbolic_trace(model)
    node_map = build_node_map(model_graph)
    named_mods = dict(model_graph.named_modules())
    add_node_pairs_map = build_add_pairs(model_graph, layers, node_map)
    
    input_shape = None
    for idx, name in enumerate(layers):
        if name in model.state_dict():
            W = model.state_dict()[name]
            filter_id = get_partition_from_code(configs['data_code'], W.shape[0], num_partitions, budgets[idx])
            
            if name not in add_node_pairs_map:
                for node in model_graph.graph.nodes:
                    if node_map.get(node) == name:
                        predecessor = backtrack_to_layer(node, named_mods)
                        parent = node_map.get(predecessor) if predecessor else None
                        if parent is None:
                            parent = 'inputs'  # Store the first conv layer
                            input_shape = W.shape[1]
                partition[name] = {
                    'filter_id': filter_id,
                    'parents': [parent],
                    'budget': budgets[idx]
                }
            else:
                partition[name] = {
                    'filter_id': filter_id,
                    'parents': add_node_pairs_map[name],
                    'budget': budgets[idx],
                }
            print(f"Partition information of {name}:\n", partition[name])
                
    # Ensure inputs node is included
    partition['inputs'] = {'filter_id': 
                           get_partition_from_code(configs['data_code'], input_shape, num_partitions, input_part),
                           'input_partition': input_part.tolist()} 
    print(f"Partition information of inputs:\n", partition['inputs'])
    
    partition['model_graph'] = {
        'graph': model_graph,
        'node_map': node_map,
        'addition_nodes': add_node_pairs_map,
    }
    
    configs['partition'] = partition
    return configs
    

def partition_generator(configs, model):
    partition = {}
    num_partition = {}
    # print(configs['num_partition'])
        
    # # get # partition for each layer
    # if isinstance(configs['num_partition'], int):
    #     # Todo: automatically set bn_partition
    #     ratio_partition, map_partition = {}, {}
    #     bn_partition = [int(configs['num_partition'])] * 9
    #     num = int(configs['num_partition'])
    #     maps = np.ones((num,num))
    #     for name, W in itertools.chain(model.named_parameters() , list({'inputs':None}.items())):
    #         print(name)
    #         try:
    #             print(W.size())
    #         except:
    #             pass
    #         if name=='inputs' or (len(W.size()) == 4) or (len(W.size()) == 2):
    #             if not 'out' in name:                  
    #                 num_partition[name] = num
    #                 ratio_partition[name] = [1]*num
    #                 np.fill_diagonal(maps, 0)
    #                 map_partition[name] = maps.astype(int).tolist()
                
    # elif os.path.exists(configs['num_partition']):
    with open(configs['partition_path'], "r") as stream:
        raw_dict = yaml.safe_load(stream)
        
        bn_partition = raw_dict['bn_partitions']
        ratio_partition = raw_dict['partitions']
        map_partition = raw_dict['maps']
        
        # print(ratio_partition,map_partition)
        for name, key in ratio_partition.items():
            # print('name:', name)
            # print('key:', key)
            # print(name, key)
            # ratio_partition[name] = key[0]
            ratio_partition[name] = key
            # num_partition[name] = len(key[0])
            num_partition[name] = len(key)

    # else:
    #     raise Exception("num_partition must be either a filepath or an integer")
    
    print('num_partition:', num_partition)
    print('ratio_partition:', ratio_partition)
    print('map_partition:', map_partition)
    print('bn_partition:', bn_partition)
    
    # setup bn_partition
    partition['bn_partition'] = bn_partition
    
    # setup prune ratio
    pr, configs['prune_ratio'] = configs['prune_ratio'], {}
    for name, num in num_partition.items():
        ratio = ratio_partition[name]
        # configs['prune_ratio'][name] = 1 - sum(r**2 for r in ratio) / sum(ratio)**2
        configs['prune_ratio'][name] = pr
        
    # setup partition
    ratio_prev = ratio_partition['inputs'] # for current channel ratio, which is equal to the previous filter ratio
    
    for name, W in model.named_parameters():
        if name in num_partition and num_partition[name] > 1:
            num, maps = num_partition[name], map_partition[name]
            
            # setup selected kernel ids
            filter_id  = get_partition_from_code(configs['data_code'], W.shape[0], ratio_partition[name])
            channel_id = get_partition_from_code(configs['data_code'], W.shape[1], ratio_prev)
            ratio_prev = ratio_partition[name]
            partition[name] = {'num': num, 
                               'filter_id': filter_id,
                               'channel_id': channel_id,
                               'maps': maps}
            
    # print('partition:', partition)
            
            # v0: split by 'jump'
            #own_state[name].copy_(param_s[i::num_partition])
            # v1: split by equal 'cut'
            #len_in, len_out = int(weight.shape[0]/num_partition), int(weight.shape[1]/num_partition)
            #cost_mask[i*len_in:(i+1)*len_in,i*len_out:(i+1)*len_out,:,:] = 0
            # v2: split by app-equal 'cut'
            #k1, m1 = divmod(weight.shape[0], num)
            #k2, m2 = divmod(weight.shape[1], num)
            #for i in range(num):
            #    filter_id.append(np.array(range(ParCalculator(i,k1,m1),ParCalculator(i+1,k1,m1))))
            #    channel_id.append(np.array(range(ParCalculator(i,k2,m2),ParCalculator(i+1,k2,m2))))
            
    configs['partition'] = partition    
    return configs

def get_partition_from_code(dataset, shape, num_partitions, budget):
    """
    Distributes `shape` filters across `num_partitions` based on the given `budget`.

    Args:
        dataset (str): The dataset (not used in this function but kept for compatibility).
        shape (int): The total number of filters (or neurons).
        num_partitions (int): The number of partitions (machines).
        budget (list of float): A list of size `num_partitions` where each entry represents 
                                the proportion of filters assigned to that partition.
                                Must sum to 1.

    Returns:
        list of np.ndarray: A list where each entry is a NumPy array containing 
                            the indices of the filters assigned to that partition.
    """
    if not np.isclose(sum(budget), 1.0):
        raise ValueError(f"Budget proportions must sum to 1, but got {sum(budget):.6f}.")

    p_range = np.array(range(shape))  # Index range of filters
    p_id = []
    
    # Compute absolute filter allocation per partition with rounding
    absolute_counts = np.round(np.array(budget) * shape).astype(int)

    # Ensure total allocation sums to `shape` (adjust rounding issues)
    while absolute_counts.sum() < shape:
        absolute_counts[np.argmax(budget)] += 1  # Add remaining filters to the largest budget
    while absolute_counts.sum() > shape:
        absolute_counts[np.argmin(budget)] -= 1  # Remove excess filters from the smallest budget

    # Assign filters to partitions
    start = 0
    for count in absolute_counts:
        p_id.append(np.array(p_range[start:start + count], dtype=int))
        start += count

    # Final check to ensure all filters are assigned correctly
    assigned_channels = sum(len(part) for part in p_id)
    if assigned_channels != shape:
        raise ValueError(f"Partitioning failed: {assigned_channels} filters assigned, but expected {shape}.")

    return p_id

def get_partition_from_code_legacy(dataset, shape, ratio):
    p_id = []
    #if dataset == 'flash':
    #    p_len = [64, 256, 512]
    #    p_ratio = np.cumsum([0.0]+[x/sum(p_len) for x in p_len])
    #else:
    #    p_ratio = np.cumsum([0.0]+[1/num for _ in range(num)])
    p_ratio = np.cumsum([0.0]+[x/sum(ratio) for x in ratio])
        
    p_range = np.array(range(shape))
    for i in range(len(ratio)):
        p_id.append(p_range[int(p_ratio[i]*shape):int(p_ratio[i+1]*shape)])   
    
    # Check if there are remaining elements and add them to the last list
    last_index = int(p_ratio[-1] * shape)
    if last_index < shape:
        p_id[-1] = np.concatenate((p_id[-1], p_range[last_index:]))
    
    return p_id
                              
def ParCalculator(i,k,m):
    return i*k+min(i, m)

def set_communication_cost(model, partition):
    """
    Computes communication cost masks based on filter assignments.

    - Uses `filter_id` of parent layers instead of `channel_id`.
    - If the parent is `'inputs'`, it is ignored.
    - Computes communication cost for each layer and stores it in `comm_costs`.

    Args:
        model (torch.nn.Module): The model containing the parameters.
        partition (dict): The partitioning information, including filter assignments.

    Returns:
        dict: `comm_costs` mapping layer names to their respective communication cost tensors.
    """
    comm_costs = {}
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
                        if len(parent_filter_ids[j]) > 0:
                            cost_mask[partition[name]['filter_id'][i][:, None], parent_filter_ids[j]] = maps

            comm_costs[name] = torch.from_numpy(cost_mask).to(device)

    return comm_costs

def compute_comm_cost(model, partition):
    """
    Computes the communication cost of the model while accounting for the fact that 
    if a machine is already computing an output filter with one input channel, 
    additional input channels for the same output filter are free.

    Args:
        model (torch.nn.Module): The trained model.
        partition (dict): The partitioning information.

    Returns:
        float: The total communication cost of the model.
    """

    total_comm_cost = 0
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

            # Compute the absolute sum across kernel dimensions
            if is_conv:
                W_flat = np.sum(np.abs(weight.reshape(shape[0], shape[1], -1)), axis=2)
            else:  # Fully connected layer
                W_flat = np.abs(weight)

            # Iterate through all partitions and compute communication cost
            for n in range(num_machines):  # Output filter partitions
                for C_out in layer_partition['filter_id'][n]:  # Output filters in this partition
                    for parent_layer in parents:
                        if parent_layer not in partition:
                            raise ValueError(f"Parent layer {parent_layer} not found in partition dictionary.")
                        parent_filters = partition[parent_layer]['filter_id']

                        for i in range(num_machines):  # Input partitions
                            if i == n:
                                continue  # Skip if input and output are on the same machine

                            input_channels = parent_filters[i]

                            # Check if weights between this output filter and input channels are nonzero
                            if np.any(W_flat[C_out, input_channels]):
                                total_comm_cost += comm_cost_map[n][i] * outsize

    return total_comm_cost

def featuremap_summary(model, partition, inputs):
    '''
    Calculate the size of output (feature map) of each layer
    '''
    def register_hook(name):
        def hook(module, input, output):
            outshape = list(output.size())
            outsize = outshape[2]*outshape[3] if len(outshape)==4 else 1 # ignores dim 0 and 1 which are for batch size and channel size?
            partition[name]['outsize'] = outsize
            #print(name, class_name, outsize, outshape)
        return hook
    
    for name, layer in model.named_modules():
    #     print(name)
    #     print(layer)
        name = name+'.weight'
        if name in partition:
            layer.register_forward_hook(register_hook(name))
    
    model(*inputs)
    
    start_time = time.time()
    model(*inputs)
    print('Inference time per data is {:.6f}ms.'.format((time.time()-start_time)*1000))
    
    # for last layer before prediction layer
    for name, W in model.named_parameters():
        if name in partition and 'outsize' not in partition[name]: 
            partition[name]['outsize'] = 1
        if name in partition: print(name, partition[name]['outsize'])
            
    return partition

def masknet_to_dense(masknet, model):
    device = next(model.parameters()).device
    own_state = model.state_dict()
    
    # load dense variables
    for (name, W) in masknet.named_parameters():
        if "mask" not in name:
            own_state[name].copy_(W.data)

    # update dense variables
    for (name, W) in masknet.named_parameters():
        if "mask" in name:
            W_d = own_state[name.replace('mask', "weight")]
            
            weight = W.cpu().detach().numpy()
            weight_d = W_d.cpu().detach().numpy()
            
            own_state[name.replace('mask', "weight")].copy_(torch.from_numpy(weight * weight_d))
            
def get_model_mask(model):
    masks = {}
    device = next(model.parameters()).device
    
    for name, W in (model.named_parameters()):
        if not W.requires_grad:
            continue
        weight = W.cpu().detach().numpy()
        non_zeros = (weight != 0)
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros)
        W = torch.from_numpy(weight).to(device)
        W.data = W
        masks[name] = zero_mask.to(device)
        #print(name,zero_mask.nonzero().shape)
    return masks

def set_trainable_mask(model, requires_grad=False, target='weight'):
    for name, W in (model.named_parameters()):
        if target in name:
            W.requires_grad = requires_grad
            
            
############################################################
#  Graph Analysis: Pre-Scan for Add Pairs
############################################################

def backtrack_to_layer(start_node, named_mods):
    """
    From 'start_node', keep going backwards in the graph until we find the first
    call_module node that is an nn.Conv2d or nn.Linear (and thus appears in our partition).
    This accounts for BN/ReLU etc. in between.
    
    If no convolution is found, return None.
    """
    visited = set()
    stack = [arg for arg in start_node.args if isinstance(arg, fx.Node)]
    
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        
        if cur.op == "call_module":
            submod = named_mods.get(cur.target, None)
            if isinstance(submod, (nn.Conv2d, nn.Linear)):
                return cur  # Return first found conv node
        
        for arg in cur.args:
            if isinstance(arg, fx.Node):
                stack.append(arg)
    
    return None


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

def find_next_layer(start_node, named_mods):
    """
    Traverse forward from 'start_node' to find the next nn.Conv2d or nn.Linear node.
    If no convolution is found, return None.
    """
    visited = set()
    queue = [start_node]
    
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        
        for user in cur.users:
            submod = named_mods.get(user.target, None)
            if user.op == "call_module" and isinstance(submod, (nn.Conv2d, nn.Linear)):
                return user  # Return first found conv node
            queue.append(user)
    
    return None

def build_add_pairs(gm, layer_names, node_map):
    """
    We do a pre-scan of the graph to find any add nodes and identify exactly
    which two layer nodes feed them. We'll store them in a map:
       layer_add_node -> (layerA, layerB)
    """
    add_pairs = {}  # add_node -> (layerA, layerB)
    named_mods = dict(gm.named_modules())
    
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in (operator.add, torch.add):
            # look for 2-arg add
            if len(node.args) != 2:
                continue
            lhs, rhs = node.args
            if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
                continue

            # backtrack each side to find layer(s)
            lhs_layer = backtrack_to_layer(lhs, named_mods)
            rhs_layer = backtrack_to_layer(rhs, named_mods)
            
            # If exactly 1 layer on each side, check if they are in partition_dict
            if lhs_layer and rhs_layer:
                layerA = node_map.get(lhs_layer, None)
                layerB = node_map.get(rhs_layer, None)
            # must be in layer_names
            if layerA and layerB and layerA in layer_names and layerB in layer_names:
                # Find the next layer after this addition
                layer_after_add = find_next_layer(node, named_mods)
                if layer_after_add:
                    layer_after_layer = node_map.get(layer_after_add, None)
                    if layer_after_layer:
                        add_pairs[layer_after_layer] = [layerA, layerB]

    return add_pairs


def build_node_map(gm):
    """
    Return a dict: node -> 'submodule_name.weight'
    if node is a call_module that might correspond to layer_names keys.
    """
    nmap = {}
    for n in gm.graph.nodes:
        if n.op == "call_module":
            submod_name = n.target
            candidate_key = submod_name + ".weight"
            nmap[n] = candidate_key
    return nmap
