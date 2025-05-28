from __future__ import print_function
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import operator
from numpy import linalg as LA
import numpy as np
import yaml
import random
from itertools import combinations
import collections

from ..utils.testers import *



class ADMM:
    def __init__(self, config_dict, model, rho=0.001, target='weight'):
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = rho
        self.target = target
        
        self.rhos = {}
        self.prune_ratio = config_dict['prune_ratio']
        self.device = config_dict['device']
        self.par_first_layer = config_dict['par_first_layer']
        self.partition = config_dict['partition']
        self.init(model)

    def init(self, model):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file

        """
        # setup pruning ratio
        self.prune_ratios = {}
        #counter = 0
        for name, weight in model.named_parameters():
            if name in self.partition:
                #counter += 1
                #if not self.par_first_layer and counter==1: continue
                self.prune_ratios[name] = self.prune_ratio
        
        # setup rho
        for k, v in self.prune_ratios.items():
            self.rhos[k] = self.rho
        
        print(self.prune_ratios) 
        
        # initialize aux and dual params
        for (name, W) in model.named_parameters():
            if name not in self.prune_ratios:
                continue
            self.ADMM_U[name] = torch.zeros(W.shape).to(self.device)  # add U
            self.ADMM_Z[name] = torch.Tensor(W.shape).to(self.device)  # add Z


def random_pruning(weight, prune_ratio, sparsity_type):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0],
                                   int(shape2d[0] * prune_ratio),
                                   replace=False)
        weight2d[indices, :] = 0
        weight = weight2d.reshape(shape)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = i not in indices
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(
            expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise Exception("not implemented yet")


def L1_pruning(weight, prune_ratio, sparsity_type):
    """
    projected gradient descent for comparison

    """
    percent = prune_ratio * 100
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l1_norm = LA.norm(weight2d, 1, axis=1)
    percentile = np.percentile(row_l1_norm, percent)
    under_threshold = row_l1_norm < percentile
    above_threshold = row_l1_norm > percentile
    weight2d[under_threshold, :] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    weight = weight.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(
        weight).cuda()


def weight_pruning(weight, name, prune_ratio, sparsity_type, cross_x=4, cross_f=1, partition=''):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """
    device = weight.device
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    #cross_x = args.cross_x
    #cross_f = args.cross_f
    #percent = prune_ratio * 100 * args.ratioexp
    percent = prune_ratio * 100
    
    parents = partition[name].get('parents', [])
    for parent in parents: 
        if parent not in partition:
            raise ValueError(f"Parent layer {parent} not found in partition dictionary.")

    
    if (sparsity_type == "irregular"):
        weight_temp = np.abs(
            weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp,
                                   percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32
        )  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).to(device), torch.from_numpy(weight).to(device)

    ####################################

    elif (sparsity_type == "column"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm < percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)
        return torch.from_numpy(
            expand_above_threshold).to(device), torch.from_numpy(weight).to(device)
    
    elif (sparsity_type == "row"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)  # Compute L2 norms for rows
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm < percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0  # Set rows below the threshold to zero
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)
        return torch.from_numpy(
            expand_above_threshold).to(device), torch.from_numpy(weight).to(device)

    elif (sparsity_type == 'partition'):
        num_partition = partition['num']
        shape = weight.shape
        weight3d = weight.reshape(shape[0], shape[1], -1)
        zero3d = np.zeros(shape).reshape(shape[0],shape[1], -1)
        shape3d = weight3d.shape
        if len(shape3d) == 2:
            weight3d, zero3d = weight3d[:,:,None], zero3d[:,:,None]
            
        for i in range(num_partition):
            #weight_copy[i::num_partition,i::num_partition,:,:] = weight[i::num_partition,i::num_partition,:,:]
            zero3d[partition[name]['filter_id'][i][:,None],partition[name]['channel_id'][i],:] = weight3d[partition[name]['filter_id'][i][:,None],partition[name]['channel_id'][i],:] 
        
        weight = zero3d.reshape(shape)
        return num_partition, torch.from_numpy(weight).float().to(device)    

    elif sparsity_type == "partition_row":
        num_partitions = partition['num']
        shape = weight.shape
        weight3d = weight.reshape(shape[0], shape[1], -1)  # Reshape to (out_channels, in_channels, kernel_size_prod)

        all_norms = []
        all_indices = []  # To track which rows (output filters) correspond to which (src, dst) pairs

        # First pass: Collect all L2 norms and their respective indices
        for dst in range(num_partitions):  # Destination machines handling output filters
            output_filters = partition[name]['filter_id'][dst]  # Output filters at this machine

            for src in range(num_partitions):  # Source machines providing input channels
                if src == dst:
                    continue  # Skip intra-machine connections

                # Gather input channels from all parents for the source machine
                if len(parents) == 1:
                    input_channels = partition[parents[0]]['filter_id'][src]
                else:
                    input_channels = np.concatenate([partition[parent]['filter_id'][src] for parent in parents])
                    input_channels = np.unique(input_channels)  # Remove duplicates

                if len(output_filters) == 0 or len(input_channels) == 0:
                    continue  # Skip if no valid filters or inputs in this partition

                # Extract the relevant submatrix of weights
                submatrix = weight3d[np.ix_(output_filters, input_channels)]
                row_l2_norm = LA.norm(submatrix.reshape(submatrix.shape[0], -1), axis=1)

                # Store norms and corresponding indices for later pruning
                for i, norm_val in enumerate(row_l2_norm):
                    out_idx = output_filters[i]
                    all_norms.append(norm_val)
                    all_indices.append((dst, src, out_idx, input_channels))

        if len(all_norms) == 0:
            return  # Nothing to prune

        # Compute the global pruning threshold
        global_threshold = np.percentile(all_norms, percent)

        # Second pass: Apply pruning based on the global threshold
        for norm_val, (dst, src, out_idx, input_channels) in zip(all_norms, all_indices):
            if norm_val < global_threshold:
                # Zero out the weights for this (dst, src) machine-to-machine connection
                weight3d[np.ix_([out_idx], input_channels)] = 0
                
        return num_partitions, torch.from_numpy(weight3d.reshape(shape)).to(device)
    
    elif sparsity_type == "partition_row_old":
        num_partitions = partition['num']
        shape = weight.shape
        weight3d = weight.reshape(shape[0], shape[1], -1)  # Preserve (out_channels, in_channels)

        for dst in range(num_partitions):  # Destination machines handling output filters
            output_filters = partition[name]['filter_id'][dst]  # Output filters at this machine

            for src in range(num_partitions):  # Source machines providing input channels
                if src == dst:
                    continue  # Skip intra-machine connections

                # Gather input channels from all parents of this layer for src machine
                input_channels = []
                if len(parents) == 1:
                    input_channels = partition[parents[0]]['filter_id'][src]  # Directly use the array
                else:
                    input_channels = np.concatenate([partition[parent]['filter_id'][src] for parent in parents])
                    input_channels = np.unique(input_channels)  # Remove duplicates
                
                if len(output_filters) == 0 or len(input_channels) == 0:
                    continue  # Skip if no filters or inputs in this partition

                # Compute L2 norms for only (output filter, input channel) pairs in this (src -> dst) relation
                submatrix = weight3d[np.ix_(output_filters, input_channels)]  # Extract relevant connections
                row_l2_norm = LA.norm(submatrix, axis=1)  # Compute per-output filter norm

                if row_l2_norm.size > 0:
                    percentile = np.percentile(row_l2_norm, percent)  # Get threshold
                    under_threshold = row_l2_norm < percentile  # Identify pruned connections

                    # Zero out connections **only for src's input channels** at the selected output filters
                    weight3d[np.ix_(output_filters, input_channels)] = np.where(under_threshold[:, None], 0, submatrix)

        return num_partitions, torch.from_numpy(weight3d.reshape(shape)).to(device)
    
    elif (sparsity_type == 'kernel'):
        shape = weight.shape
        weight3d = weight.reshape(shape[0], shape[1], -1)
        shape3d = weight3d.shape
        if len(shape3d) == 2:
            weight3d = weight3d[:,:,None]
        kernel_l2_norm = LA.norm(weight3d, 2, axis=2)
        percentile = np.percentile(kernel_l2_norm, percent)
        under_threshold = kernel_l2_norm <= percentile
        above_threshold = kernel_l2_norm > percentile
        weight3d[under_threshold, :] = 0
        
        weight = weight3d.reshape(shape)
        return above_threshold, torch.from_numpy(weight).to(device)
    
    elif (sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm <= percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        # weight2d[weight2d < 1e-40] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(
            expand_above_threshold).to(device), torch.from_numpy(weight).to(device)
    elif (sparsity_type == "bn_filter"):
        ## bn pruning is very similar to bias pruning
        weight_temp = np.abs(weight)
        percentile = np.percentile(weight_temp, percent)
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32
        )  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).to(device), torch.from_numpy(weight).to(device)
    else:
        raise SyntaxError("Unknown sparsity type")


def hard_prune(ADMM, model, sparsity_type, option=None, cross_x=4, cross_f=1):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda

    """

    print("hard pruning")
    for (name, W) in model.named_parameters():
        if name not in ADMM.prune_ratios:  # ignore layers that do not have rho
            continue
        cuda_pruned_weights = None
        if option == None:
            _, cuda_pruned_weights = weight_pruning(
                W, name, ADMM.prune_ratios[name], sparsity_type, cross_x,
                cross_f, ADMM.partition)  # get sparse model in cuda

        elif option == "random":
            _, cuda_pruned_weights = random_pruning(W,ADMM.prune_ratios[name],sparsity_type)

        elif option == "l1":
            _, cuda_pruned_weights = L1_pruning(W,ADMM.prune_ratios[name],sparsity_type)
        else:
            raise Exception("not implemented yet")
        W.data = cuda_pruned_weights  # replace the data field in variable


def admm_initialization(config_dict, ADMM, model, cross_x=4, cross_f=1):
    sparsity_type = config_dict['sparsity_type']
    for i, (name, W) in enumerate(model.named_parameters()):
        if name in ADMM.prune_ratios:
            _, updated_Z = weight_pruning(
                W, name, ADMM.prune_ratios[name], sparsity_type, cross_x,
                cross_f, ADMM.partition)  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
            ADMM.ADMM_Z[name] = updated_Z


def z_u_update(config_dict,
               ADMM,
               model,
               epoch,
               batch_idx,
               writer=False,
               cross_x=4,
               cross_f=1):
    
    admm_epochs, sparsity_type = config_dict['admm_epochs'], config_dict['sparsity_type']
    if epoch != 1 and (epoch - 1) % admm_epochs == 0 and batch_idx == 0:
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue
            Z_prev = None
            
            if config_dict['multi_rho']:
                 admm_multi_rho_scheduler(ADMM,name) # call multi rho scheduler every admm update
            
            ADMM.ADMM_Z[name] = W.detach() + ADMM.ADMM_U[name].detach()  # Z(k+1) = W(k+1)+U[k]

            _, updated_Z = weight_pruning(ADMM.ADMM_Z[name], name, ADMM.prune_ratios[name], sparsity_type, 
                cross_x, cross_f, ADMM.partition)  # equivalent to Euclidean Projection
            ADMM.ADMM_Z[name] = updated_Z
            ADMM.ADMM_U[name] = W.detach() - ADMM.ADMM_Z[name].detach() + ADMM.ADMM_U[name].detach()  # U(k+1) = W(k+1) - Z(k+1) +U(k)


def append_admm_loss(ADMM, model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    

    for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
        if name not in ADMM.prune_ratios:
            continue

        admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2)**2)
        #print(name, admm_loss[name], ADMM.rhos[name])
        #print(name, torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2)**2)
        # admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)  # test if Z,U are net detached
    
    mixed_loss = 0
    mixed_loss += ce_loss
    total_admm_loss = 0
    for k, v in admm_loss.items():
        mixed_loss += v
        total_admm_loss += v
    #print('admm_loss: ', mixed_loss)
    return ce_loss, total_admm_loss, mixed_loss


def admm_multi_rho_scheduler(ADMM, name):
    """
    It works better to make rho monotonically increasing
    rho: using 1.1: 
           0.01   ->  50epochs -> 1
           0.0001 -> 100epochs -> 1
         using 1.2:
           0.01   -> 25epochs -> 1
           0.0001 -> 50epochs -> 1
         using 1.3:
           0.001   -> 25epochs -> 1
         using 1.6:
           0.001   -> 16epochs -> 1
    """
    current_rho = ADMM.rhos[name]
    ADMM.rhos[name] = min(1, 1.1*current_rho)  # choose whatever you like
    
def admm_adjust_learning_rate(optimizer, epoch, config_dict):
    """ (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default 
    admm epoch is 9)

    """
    admm_epoch = config_dict['admm_epochs']
    lr = None
    if (epoch - 1) % admm_epoch == 0:
        lr = config_dict['learning_rate']
    else:
        admm_epoch_offset = (epoch - 1) % admm_epoch

        admm_step = admm_epoch / (3/2)  # roughly every 2/3 admm_epoch.
        # admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.

        lr = config_dict['learning_rate'] * (0.1**(admm_epoch_offset // admm_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

import numpy as np

def compute_partition_convergence(prev_P, partition):
    """
    Computes convergence for partition filter assignments (P) by comparing
    the previous filter assignments (prev_P) with the current ones.

    This measures the number of filters that changed machines across all partitions.

    Args:
        prev_P (dict): Previous partition assignments {'layer_name': [[filters for each machine]]}.
        partition (dict): Current partition assignments {'layer_name': {'filter_id': [[filters per machine]]}}.

    Returns:
        float: The total number of filters that changed assignments.
    """
    convergence_P = 0

    for name in prev_P:
        if name in partition:
            prev_filters = prev_P[name]  # List of lists (each list is a machine)
            curr_filters = partition[name]['filter_id']  # List of lists (each list is a machine)

            # Ensure both have the same number of partitions (machines)
            if len(prev_filters) != len(curr_filters):
                raise ValueError(f"Mismatch in partition sizes for {name}: {len(prev_filters)} vs {len(curr_filters)}")

            # Compute filter changes per machine
            for prev_set, curr_set in zip(prev_filters, curr_filters):
                prev_set, curr_set = set(prev_set), set(curr_set)
                changes = len(prev_set.symmetric_difference(curr_set))  # Count changed filters
                convergence_P += changes  # Sum changes across machines

    return convergence_P


