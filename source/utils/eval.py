import torch
import wandb
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

class AverageMeter(object):
    """Basic meter"""
    def __init__(self):
        self.reset()

    def reset(self):
        """ reset meter
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ incremental meter
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EvalHelper():
    def __init__(self, data_code):
        self.data_code = data_code
    
    def call(self, output, target):
        if self.data_code == 'flash':
            acc = [(torch.argmax(target, axis=1) == torch.argmax(output, axis=1)).sum()*100/target.size(0)]
        else:
            acc = self.accuracy(output, target, topk=(1,))
        return acc
    
    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def get_accuracy(self, model, dataloader, criterion, cepoch):
        """ Computes the precision@k for the specified values of k
            https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        losses = AverageMeter()
        top1 = AverageMeter()
        device = next(model.parameters()).device

        # for binary case
        output_all, target_all = [], []
        softmax = torch.nn.Softmax(dim=1)
        
        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                data   = ()
                for piece in batch[:-1]:
                    # print(piece.shape)
                    data += (piece.float().to(device),)
                target = batch[-1].to(device)
                # if self.data_code == 'esc':
                #     target = target.unsqueeze(1).float()

                # Concatenate the input data, it is tuple of tensors
                data = (torch.cat(data, dim=1),)


                # compute output
                output = model(*data)
                # print(output.shape, target.shape)
                # print(output, target)
                loss = criterion(output, target)

                # probabilities = torch.softmax(output, dim=1)
                # print(probabilities)
                # print(loss)

                # measure accuracy and record loss
                acc1 = self.call(output, target)
                losses.update(loss.item(), target.size(0))
                top1.update(acc1[0].item(), target.size(0))
                
                # only measured for binary classification
                if self.data_code == 'esc':
                    output = softmax(output).cpu().detach().numpy()
                    target = target.cpu().detach().numpy()
                    output_all = np.vstack((output_all,output)) if len(output_all) else output
                    target_all = np.vstack((target_all,target[:,None])) if len(target_all) else target[:,None]
                    
        print("Epoch-[{:03d}]: Test loss: {:.2f}, acc: {:.2f}.".format(cepoch, losses.avg, top1.avg,))
        
        if self.data_code == 'esc':
            auc = roc_auc_score(target_all, output_all[:, 1])
            # auc = roc_auc_score(target_all, output_all.squeeze())
            prec, recall, fscore, _ = precision_recall_fscore_support(target_all, np.argmax(output_all, axis=1), average='macro')
            print("prec: {:.4f}, recall: {:.4f}, auc: {:.4f}".format(prec, recall, auc))
        

        return losses.avg, top1.avg

def format_metric(value, suffix=""):
    """Formats a metric if it's numeric, otherwise returns it as is."""
    return f"{value:.4f}{suffix}" if isinstance(value, (int, float)) else value
    
class ExperimentLogger:
    def __init__(self, experiment_dir, use_wandb=False):
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb
        os.makedirs(experiment_dir, exist_ok=True)
        
        self.logs = {
            'epoch': [],
            'global_loss': [],
            'train_acc': [],
            'ML_loss': [],
            'comm_loss': [],
            'ADMM_loss': [],
            'agreement_quality': [],  # ||Z - W||
            'convergence_W': [],  # ||W_t - W_{t-1}||
            'convergence_Z': [],  # ||Z_t - Z_{t-1}||
            'convergence_P': [],  # ||P_t - P_{t-1}||
            'comm_cost': [],  # Communication cost over training
            'constraint_sparsity_W': [],
            'constraint_sparsity_Z': [],
            'partition_validity': [],
            'test_loss': [], 
            'test_acc': [],
            'P_costs': [],
            'elapsed_time': [],
        }

    def log(self, epoch, **kwargs):
        self.logs['epoch'].append(epoch)
        P_avg = 'N/A'
        for key, value in kwargs.items():
            if key == 'P_costs':
                if value != 'N/A' and len(value) > 0:
                    self.logs[key].extend(value)
                    P_avg = np.mean(value)
            else:
                self.logs[key].append(value)
        
        # Print metrics at the end of each epoch
        print(f"\n📌 **Epoch {epoch} Summary**:")
        print(f"    ➤ Global Loss: {format_metric(kwargs.get('global_loss', 'N/A'))}")
        print(f"    ➤ Train Accuracy: {format_metric(kwargs.get('train_acc', 'N/A'), '%')}")
        print(f"    ➤ ML Loss: {format_metric(kwargs.get('ML_loss', 'N/A'))}")
        print(f"    ➤ Comm Loss: {format_metric(kwargs.get('comm_loss', 'N/A'))}")
        print(f"    ➤ P Average Cost: {format_metric(P_avg)}")
        print(f"    ➤ ADMM Loss: {format_metric(kwargs.get('ADMM_loss', 'N/A'))}")
        print(f"    ➤ Agreement Quality: {format_metric(kwargs.get('agreement_quality', 'N/A'))}")
        print(f"    ➤ Convergence W: {format_metric(kwargs.get('convergence_W', 'N/A'))}")
        print(f"    ➤ Convergence Z: {format_metric(kwargs.get('convergence_Z', 'N/A'))}")
        print(f"    ➤ Convergence P: {format_metric(kwargs.get('convergence_P', 'N/A'))}")
        print(f"    ➤ Communication Cost: {format_metric(kwargs.get('comm_cost', 'N/A'))}")
        print(f"    ➤ Sparsity W: {format_metric(kwargs.get('constraint_sparsity_W', 'N/A'))}")
        print(f"    ➤ Sparsity Z: {format_metric(kwargs.get('constraint_sparsity_Z', 'N/A'))}")
        print(f"    ➤ Partition Validity: {'✔️' if kwargs.get('partition_validity', False) else '❌'}")
        print(f"    ➤ Test Loss: {format_metric(kwargs.get('test_loss', 'N/A'))}")
        print(f"    ➤ Test Accuracy: {format_metric(kwargs.get('test_acc', 'N/A'), '%')}")
        print(f"    🕒 Elapsed Time: {format_metric(kwargs.get('elapsed_time', 'N/A'), 's')}")
        
        # wandb logging
        if self.use_wandb:
            print("Logging metrics to wandb")
            log_dict = {}
            for k, v in kwargs.items():
                if v != "N/A" and k != "P_costs":
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().item()
                    log_dict[k] = v
            log_dict['epoch'] = epoch
            wandb.log(log_dict, step=epoch+1)

    def save(self):
        for key, value in self.logs.items():
            # Handle PyTorch tensors (detach, move to CPU, convert to numpy, then to list)
            self.logs[key] = [v.detach().cpu().item() if isinstance(v, torch.Tensor) else v for v in value]
            
        # Exclude 'P_costs' from logs before saving
        logs_filtered = {k: v for k, v in self.logs.items() if k != "P_costs"}

        # Convert to DataFrame and save
        df = pd.DataFrame(logs_filtered)
        df.to_csv(os.path.join(self.experiment_dir, "metrics.csv"), index=False)

    def plot(self):
        # Mapping metric keys to human-readable names
        metric_names = {
            'global_loss': 'Global Loss',
            'train_acc': 'Training Accuracy (%)',
            'ML_loss': 'ML Loss',
            'comm_loss': 'Communication Loss',
            'ADMM_loss': 'ADMM Loss',
            'agreement_quality': 'Agreement Quality',
            'convergence_W': 'Convergence (W)',
            'convergence_Z': 'Convergence (Z)',
            'convergence_P': 'Convergence (P)',
            'comm_cost': 'Communication Cost',
            'test_loss': 'Test Loss',
            'test_acc': 'Test Accuracy (%)',
            'elapsed_time': 'Elapsed Time (s)'
        }

        for key, label in metric_names.items():
            if key not in self.logs:
                print(f"Skipping plot for {key}, key not found in logs.")
                continue

            values = np.array([float(v) if v != "N/A" else np.nan for v in self.logs[key]])  # Convert to numbers, replace "N/A" with NaN

            if np.all(np.isnan(values)):  # Skip plotting if all values are NaN
                print(f"Skipping plot for {label}, all values are NaN.")
                continue

            epochs = np.array(self.logs['epoch'])

            # Ensure x and y have the same length, skipping NaN epochs
            valid_indices = ~np.isnan(values)
            epochs, values = epochs[valid_indices], values[valid_indices]

            plt.figure()
            plt.plot(epochs, values, label=label, marker='o', linestyle='-')
            plt.xlabel("Epoch")
            plt.ylabel(label)
            plt.title(f"{label} over Epochs")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(self.experiment_dir, f"{key}.png"))
            plt.close()
            
        # Special case: Handling "P_costs"
        if 'P_costs' in self.logs and len(self.logs['P_costs']) > 0:
            print("Processing P_costs...")

            # Convert P_costs into a NumPy array
            P_costs_array = np.array(self.logs['P_costs'])

            # Ensure it's a 1D list before plotting
            if P_costs_array.ndim == 1:
                plt.figure(figsize=(10, 5))
                plt.plot(range(len(P_costs_array)), P_costs_array, label="P_costs", linestyle='-', marker='.')
                plt.xlabel("Batch Index")
                plt.ylabel("P Matrix Cost")
                plt.title("P Assignment Cost over Training")
                plt.legend()
                plt.grid()
                plt.savefig(os.path.join(self.experiment_dir, "P_costs.png"))
                plt.close()

            else:
                print("Unexpected format for P_costs, skipping plot.")
