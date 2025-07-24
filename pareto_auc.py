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

from torcheval.metrics import MulticlassAUROC

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


def compute_roc_auc(model, test_loader, data_code, device):
    model = model.to(device)
    
    model.eval()

    outx = []
    labelx = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            control=False   # True: run for smaller dataset
            if batch_idx > 5 and control:
                print("!!!! Running for SMALL dataset !!!!")
                break
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            #print(outputs.shape, labels.shape)
            outx.append(outputs)
            labelx.append(labels)
    
    voutx = torch.vstack(outx)
    vxx = torch.cat(labelx, dim=0)
    vlabelx = torch.flatten(vxx)  

    if data_code == "cifar10":
        metric = MulticlassAUROC(num_classes=10)
    elif data_code == "cifar100":
        metric = MulticlassAUROC(num_classes=100)

    metric.update(voutx, vlabelx)
    roc_auc = metric.compute()    
    print("Computed AUC: ", roc_auc)

    return round(roc_auc.item(),4)
    

def evaluate_roc_auc(experiment_logs, test_loader, data_code, device, seed=1234):

    # Set all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
   
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
        '''
        partition_data = load_partition_data(folder_path)
        if not partition_data:
            print(f"SKIPPING FOR FOLDER: {folder}, partition_data")
            continue
        '''
        
        if experiment_details["model"] == "resnet18":
            model = models.__dict__[experiment_details['model']](nn.Conv2d, nn.BatchNorm2d, num_classes=10)
        elif experiment_details["model"] == "resnet101":
            model = models.__dict__[experiment_details['model']](get_layers('regular'), get_bn_layers('regular'), num_classes=100)


        model.load_state_dict(torch.load(model_state_path, map_location=torch.device(device)))
        
        
        # @@@@ AUC
        
        best_acc_file = os.path.join(folder_path, "best_accuracy.txt")
        if not os.path.exists(best_acc_file):
            print(f"SKIPPING FOR FOLDER: {folder}, roc_auc")
            continue

        with open(best_acc_file, 'r') as f:
            match = re.search(r'Best ROC_AUC: ([0-9\.]+)', f.read())
            if match is None:
                print("ROC_AUC Not Found in best_accuracy.txy file... Start computing it!")
                roc_auc = compute_roc_auc(model, test_loader, data_code, device)
                #print(roc_auc)
                best_acc_file = os.path.join(folder_path, "best_accuracy.txt")
                with open(best_acc_file, 'a') as f:
                    f.write(f"Best ROC_AUC: {roc_auc}\n")
            else:
                roc_auc = float(match.group(1))
                print("File best_accuracy.txt contains ROC_AUC: ", roc_auc)
        

    
    
if __name__ == "__main__":
    # experiment_logs_dtelecom, experiment_logs_abilene, 
    # experiment_logs_watts_strogatz, experiment_logs_barabasi_albert
    # experiment_logs_uniform
    experiment_logs_path = "experiment_logs_uniform_10"
    data_code = "cifar10"        # valid cifar10, cifar100
    batch_size = 128
    device = "cuda"
   
    _, test_loader = get_dataset_from_code(data_code, batch_size)

    evaluate_roc_auc(experiment_logs_path, test_loader, data_code, device)
