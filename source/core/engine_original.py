from .trainer import *
from ..utils.dataset import *
from ..utils.io import *
from ..utils.eval import *
from ..utils.misc import *
from ..utils.masks import *
from ..utils.assignment import *
from ..utils.calculate_flops.calflops import calflops
from ..utils.relaxed_utils import *
from .admm import *

import time
import wandb
from torch.autograd import Variable
from torchsummary import summary
from torchviz import make_dot
import torch.onnx
import traceback
import numpy as np
import random

class MoP:
    """
    """

    def __init__(self, configs, print_model=True, print_params=True):
        
        torch.manual_seed(configs['seed'])
        np.random.seed(configs['seed'])
        random.seed(configs['seed'])
        torch.cuda.manual_seed(configs['seed'])
        torch.cuda.manual_seed_all(configs['seed'])
        self.configs = configs
        self.model_file = configs['load_pruned_model_file']
        
        # Handle dataset
        self.train_loader, self.test_loader = get_dataset_from_code(configs['data_code'], configs['batch_size'])

        # Initialize evaluation metrics
        self.evalHelper   = EvalHelper(configs['data_code'])
        
        # Load device
        self.device = configs["device"]
        
        # Create model
        self.model = get_model_from_code(configs).to(self.device)
        
        # Load pretrained weights
        if configs["load_dense_model"]:
            state_dict = torch.load(get_model_path("{}".format(configs["load_dense_model_file"])), map_location=self.device)
            self.model = load_state_dict(self.model, 
                                         state_dict['model_state_dict'] if 'model_state_dict' in state_dict 
                                         else state_dict['state_dict'] if 'state_dict' in state_dict else state_dict,)
            # Compute Accuracy
            #criterion,_,_ = set_optimizer(self.configs, self.model, self.train_loader, self.configs['optimizer'], 
            #                              self.configs['learning_rate'], self.configs['epochs'])
            #test_loss, acc = self.test_model(self.model, criterion)
            
        else:
            print('standard train')
            #self.model.apply(init_weights)
            self.train()
        
        # Print the model
        if print_model:
            print("======== MODEL INFO =========")
            print(self.model)
            print("=" * 40)

        # Print the number of parameters
        if print_params:
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"TOTAL NUMBER OF PARAMETERS = {n_parameters}")
            print("-" * 40)
        
        # Get input shape from data_code
        self.input_var = get_input_from_code(configs)

        if configs['create_partition']:
            # Create partition and save to yaml file
            create_partition(configs, self.model)
            
        # Config partitions and prune_ratio
        self.configs = generate_partition(configs, self.model)

        # Compute output size of each layer
        self.configs['partition'] = featuremap_summary(self.model, self.configs['partition'], self.input_var)

        # Setup communication costs
        self.configs['comm_costs'] = set_communication_cost(self.model, self.configs['partition'],)
        # print('Communication cost is set', self.configs['comm_costs'])

        # Calculate flops
        calflops(self.model, self.input_var)

        # Test before prune
        test_partition_with_free_channels(self.model, partition=self.configs['partition'], use_wandb=self.configs['use_wandb'], epoch=0)

        # Plot model
        # layer_id = (2,6,11,15)
        # layer_id = (2,4)
        # plot_layer(self.model, self.configs['partition'], layer_id=layer_id,
        #            savepath=get_fig_path("{}".format('.'.join(configs["load_model_file"].split('.')[:-1]))))
            
    def prune(self):
        experiment_dir = self.configs['experiment_dir'] if 'experiment_dir' in self.configs else os.path.join(self.configs['log_dir'], f"experiment_{int(time.time())}")
        if 'experiment_dir' not in self.configs: 
            self.configs['experiment_dir'] = experiment_dir
        logger = ExperimentLogger(experiment_dir, use_wandb=self.configs.get('use_wandb', False))
        start_time = time.time()
        
        if not self.configs['plot']:
            nepoch = self.configs['admm_epochs'] if self.configs['prune_ratio'] != 0 else 0
            criterion, optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                                self.configs['optimizer'], self.configs['learning_rate'], nepoch)
            
            # Initializing ADMM; if not admm, do hard pruning only
            initadmm=ADMM(self.configs, self.model, rho=self.configs['rho'])
            self.configs["new_P"]=initadmm.init_assignment(hard=True)
            admm = initadmm if self.configs['admm'] else None
            
            prev_W = {name: W.clone().detach() for name, W in self.model.named_parameters() if name in self.configs['partition']}
            prev_Z = {name: admm.ADMM_Z[name].clone().detach() if admm else None for name, W in self.model.named_parameters() if name in self.configs['partition']}
            prev_P = {name: self.configs['partition'][name]['filter_id'].copy() for name in self.configs['partition']['layers']}
            metrics = {}
            
            try:
                # prune
                for cepoch in range(0, nepoch+1):
                    if cepoch>0:
                        print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                        metrics = standard_train(self.configs, cepoch, self.model, self.train_loader, 
                                    criterion, optimizer, scheduler, ADMM=admm, comm=True, old_comm_loss=True)
                        # Compute Agreement Quality ||Z - W||
                        agreement_quality = sum(torch.norm(W - admm.ADMM_Z[name]) for name, W in self.model.named_parameters() if name in admm.ADMM_Z)

                        # Compute Convergence
                        convergence_W = sum(torch.norm(W - prev_W[name]) for name, W in self.model.named_parameters() if name in prev_W)
                        convergence_Z = sum(torch.norm(admm.ADMM_Z[name] - prev_Z[name]) for name, W in self.model.named_parameters() if name in prev_Z)
                        convergence_P = compute_partition_convergence(prev_P, self.configs['partition'])

                        convergence_W,convergence_Z,convergence_P=admm.convergence()

                    test_loss, acc = self.test_model(self.model, criterion, cepoch)
                    
                    if self.configs['reassign']:
                        save_partition(self.configs, cepoch, os.path.join(experiment_dir, f"partition_epoch_{cepoch}"))
                        if self.configs.get('use_wandb', False) and cepoch==0:
                            artifact = wandb.Artifact(name="initial_partition", type="config")
                            artifact.add_file(os.path.join(experiment_dir, f"partition_epoch_{cepoch}"))
                            wandb.log_artifact(artifact)
                            

                    

                    # Compute Communication Cost Reduction
                    #comm_cost = compute_comm_cost(self.model, self.configs['partition'])
                    comm_cost=comunication_penalty(self.model,initadmm,self.configs,self.configs["new_P"])

                    #print("comm_cost=",comm_cost)
                    

                    # Check Sparsity Constraint
                    sparsity_W = sum(torch.sum(W == 0).item() / W.numel() for name, W in self.model.named_parameters() if name in admm.ADMM_Z)
                    sparsity_Z = sum(torch.sum(admm.ADMM_Z[name] == 0).item() / admm.ADMM_Z[name].numel() for name in admm.ADMM_Z)

                    # Validate Partition
                    partition_validity = all(len(set(np.concatenate(part['filter_id']))) == len(np.concatenate(part['filter_id']))
                                             for name, part in self.configs['partition'].items() 
                                             if name in self.configs['partition']['layers'])
                                         
                                     

                    logger.log(cepoch, 
                               global_loss=metrics['batch_total_loss'].avg if cepoch > 0 else 'N/A', 
                               train_acc=metrics['batch_loss'].avg if cepoch > 0 else 'N/A', 
                               ML_loss=metrics['batch_loss'].avg if cepoch > 0 else 'N/A', 
                               comm_loss=metrics['batch_comm'].avg if cepoch > 0 else 'N/A', 
                               ADMM_loss=metrics['admm_loss'] if cepoch > 0 else 'N/A',  
                               agreement_quality=agreement_quality if cepoch > 0 else 'N/A', 
                               convergence_W=convergence_W if cepoch > 0 else 'N/A', 
                               convergence_Z=convergence_Z if cepoch > 0 else 'N/A', 
                               convergence_P=convergence_P if cepoch > 0 else 'N/A', 
                               comm_cost=comm_cost, 
                               constraint_sparsity_W=sparsity_W, 
                               constraint_sparsity_Z=sparsity_Z, 
                               partition_validity=partition_validity,
                               test_loss=test_loss,
                               test_acc=acc,
                               P_costs=metrics['P_costs'] if cepoch > 0 else 'N/A',
                               elapsed_time= time.time() - start_time
                    )

                    prev_W = {name: W.clone().detach() for name, W in self.model.named_parameters() if name in prev_W}
                    prev_Z = {name: admm.ADMM_Z[name].clone().detach() if admm else None for name, W in self.model.named_parameters() if name in prev_Z}
                    prev_P = {name: self.configs['partition'][name]['filter_id'].copy() for name in self.configs['partition']['layers']}
            
            except KeyboardInterrupt:
                print("\nTraining interrupted. Saving progress...")
            
            except Exception as e:
                print(f"\nUnexpected error encountered: {e}")
                traceback.print_exc()  # Print the full traceback for debugging
                raise  # Re-raise the error after printing for visibility
            
            # hard prune
            if self.configs['prune_ratio'] != 0:
                hard_prune(admm, self.model, self.configs['sparsity_type'], option=None)
            self.configs['comm_costs'] = set_communication_cost(self.model, self.configs['partition'],)
            if self.configs['prune_ratio'] == 0 and self.configs['reassign']:
                P_cost = update_assignments(self.model, self.configs, use_wandb=False)
                    
            save_partition(self.configs, cepoch, os.path.join(experiment_dir, "partition_final"))  # Save last partition state
            if self.configs.get('use_wandb', False):
                artifact = wandb.Artifact(name="final_partition", type="config")
                artifact.add_file(os.path.join(experiment_dir, f"partition_final"))
                wandb.log_artifact(artifact)
            torch.save(self.model.state_dict(), os.path.join(experiment_dir, "final_model.pt"))
            # test sparsity
            test_kernel_sparsity(self.model, partition=self.configs['partition'])
            test_partition_with_free_channels(self.model, partition=self.configs['partition'], use_wandb=self.configs['use_wandb'], epoch=nepoch+2)
            if self.configs['prune_ratio'] != 0:
                logger.save()
                logger.plot()
                print("Progress saved. Exiting gracefully.")

        else:
            self.model = get_model_from_code(self.configs).to(self.configs['device'])
            state_dict = torch.load(get_model_path_split("{}".format(self.configs["load_pruned_model_file"])), map_location=self.configs['device'])
            self.model = load_state_dict(self.model, 
                                            state_dict['model_state_dict'] if 'model_state_dict' in state_dict 
                                            else state_dict['state_dict'] if 'state_dict' in state_dict else state_dict,)
            


            # dot = make_dot(self.model(self.input_var), params=dict(self.model.named_parameters()))
            # dot.format = 'png'
            # dot.render(get_fig_path("{}".format('.'.join(self.model_file.split('.')[:-1]))))

            torch.onnx.export(self.model, self.input_var, get_fig_path("{}".format('.'.join(self.model_file.split('.')[:-1]))+'.onnx'))

            # plot first conv layer
            plot_layer(self.model, self.configs['partition'], layer_id=(5,),
                    savepath=get_fig_path("{}".format('.'.join(self.model_file.split('.')[:-1]))))
        #save_model(self.model, get_model_path("{}.pt".format('.'.join(self.model_file.split('.')[:-1])+'_hardprune')))
                
    def finetune(self):
        # Todo: seperate BN
        #self.parmodel = models.__dict__[self.configs['model']](self.cl, self.bn,
        #                                               num_classes=self.configs['num_classes'],
        #                                               #bn_partition=self.configs['partition']['bn_partition']
        #                                               ).to(self.device)
        
        #self.parmodel = load_state_dict(self.parmodel, 
        #                                self.model.state_dict(),
        #                                #bn_par=True, 
        #                                #partition=self.configs['partition']
        #                                )
        if not self.configs['plot']:
            print("======== MODEL INFO =========")
            print(self.model)
            print("=" * 40)
            prune_ratios = {}
            pr = self.configs['prune_ratio']
            for name, W in (self.model.named_parameters()):
                prune_ratios[name] = pr
            calflops(self.model, self.input_var, prune_ratios)
            
            # get mask
            masks = get_model_mask(model=self.model)
        
            # masked retrain
            nepoch = self.configs['retrain_ep'] if self.configs['prune_ratio'] != 0 else 0
            criterion, optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                                self.configs['retrain_opt'], self.configs['retrain_lr'], nepoch)
        
            best = 0
            try:
                for cepoch in range(0, nepoch+1):
                    if cepoch>0:
                        print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                        _ = standard_train(self.configs, cepoch, self.model, self.train_loader, 
                                    criterion, optimizer, scheduler, masks=masks, old_comm_loss=True)
                    test_loss, acc = self.test_model(self.model, criterion, cepoch)
                    if acc > best:
                        best = acc
                        save_model(self.model, os.path.join(self.configs['experiment_dir'], 'fine_tuned.pt'))
                        print('Save model')
            except KeyboardInterrupt:
                print("\nTraining interrupted. Saving progress...")
            
            # Save best accuracy in a text file
            best_acc_path = os.path.join(self.configs['experiment_dir'], 'best_accuracy.txt')
            with open(best_acc_path, 'w') as f:
                f.write(f"Best Fine-Tuned Accuracy: {best:.4f}%\n")
            
            if self.configs.get('use_wandb', False):
                wandb.log({"best_acc": best})

            print(f"✅ Best accuracy logged at: {best_acc_path}")
            
            test_kernel_sparsity(self.model, partition=self.configs['partition'])
            test_partition_with_free_channels(self.model, partition=self.configs['partition'], use_wandb=self.configs['use_wandb'], epoch=self.configs['admm_epochs']+3)
        else:
            pass
    
    def pruneMask(self):
        nepoch = self.configs['epochs']
        criterion, optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                             self.configs['optimizer'], self.configs['learning_rate'], nepoch)
        
        # Initializing ADMM; if not admm, do hard pruning only
        admm = ADMM(self.configs, self.model, rho=self.configs['rho'], target='mask') if self.configs['admm'] else None
        
        # fix weights
        set_trainable_mask(self.model, requires_grad=False, target='weight')
        
        # prune
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                _ = standard_train(self.configs, cepoch, self.model, self.train_loader, criterion, 
                               optimizer, scheduler, ADMM=admm, comm=True, old_comm_loss=True)
            test_loss, acc = self.test_model(self.model, criterion, cepoch)
            
        # hard prune
        hard_prune(admm, self.model, self.configs['sparsity_type'], option=None)
        test_filter_sparsity(self.model)
            
    def finetuneWeight(self):
        self.model_r = models.__dict__[self.configs['model']](get_layers('regular'), get_bn_layers('regular'),
                                                       num_classes=self.configs['num_classes'],
                                                       ).to(self.device)
        # transfer weight
        print('Finetune starts')
        masknet_to_dense(self.model, self.model_r)
        test_filter_sparsity(self.model_r)
        # test_partition(self.model_r, partition=self.num_partition)
        test_partition(self.model_r, partition=self.configs['partition'])
        
        # get mask
        masks = get_model_mask(model=self.model_r)
    
        # masked retrain
        nepoch = self.configs['retrain_ep']
        criterion, optimizer, scheduler = set_optimizer(self.configs, self.model_r, self.train_loader, \
                                             self.configs['retrain_opt'], self.configs['retrain_lr'], nepoch)
    
        best = 0
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                _ = standard_train(self.configs, cepoch, self.model_r, self.train_loader, 
                               criterion, optimizer, scheduler, masks=masks, old_comm_loss=True)
            test_loss, acc = self.test_model(self.model_r, criterion, cepoch)
            if acc > best:
                best = acc
                save_model(self.model_r, get_model_path("{}".format(self.model_file)))
        
        test_filter_sparsity(self.model_r)
            
    def train(self):
        nepoch = self.configs['epochs']
        criterion, optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                             self.configs['optimizer'], self.configs['learning_rate'], nepoch)
        best = 0
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                _ = standard_train(self.configs, cepoch, self.model, self.train_loader, criterion, optimizer, scheduler)
                
            test_loss, acc = self.test_model(self.model, criterion, cepoch)
            if acc > best:
                best = acc
                # save_model(self.model, get_model_path("{}".format(self.model_file.split('.')[0]+'.pt')))
                format_name = self.model_file.split('-')
                save_model(self.model, get_model_path("{}".format('-'.join(format_name[:2])+'.pt')))
                print('Save model')
                
    def test_model(self, model, criterion, cepoch=0):
        test_loss, acc = self.evalHelper.get_accuracy(model, self.test_loader, criterion, cepoch)
        return test_loss, acc
    