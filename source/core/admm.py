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
import sys
import copy
#from ..utils.relaxed_utils import *
from ..utils.testers import *

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
import sys
from ..utils.assignment import *



def translate_to_tensor(L):
    num=len(L)
    lsize=0
    for a in L:
        lsize+=len(a)
    tensor=torch.zeros((lsize,num)).float()
    for m in range(num):
        for n in L[m]:
            tensor[n,m]=1
    return tensor.to(device='cuda')

def new_to_old_partition(newP):
    m=newP.shape[1]
    L=[]
    i=newP.nonzero()
    for j in range(m):
        l=i[i[:,1]==j][:,0].tolist()
        #print(l)
        L.append(np.array(l).astype(int))
    #print(L)
    return L


class ADMM:
    def __init__(self, config_dict, model, rho=0.001, target='weight',approach="original",penalty="full"):
        self.approach=config_dict['approach'] # "original", "relaxed"
        self.penalty=config_dict['penalty'] #"aggregate_partition_rows","full"
        self.config=config_dict
        self.reassign=config_dict['reassign']
        self.sparsity_type=config_dict['sparsity_type'] # "irregular", "partition_row"
        self.model=model
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

        ######
        self.P={}
        self.layer_info={}
        self.bn_partition=self.partition['bn_partition']
        self.num=self.partition['num']
        self.C=torch.FloatTensor(self.partition['maps']).to(self.device)
        self.input_partition=self.partition['input_partition']
        self.initial_budget=self.partition['initial_budget']
        self.layer_names=self.partition['layers']
        self.model_graph=self.partition['model_graph']
        for l in self.layer_names:
            lsize=0
            for a in self.partition[l]['filter_id']:
                lsize+=len(a)
            self.layer_info[l]= {k: v for k, v in self.partition[l].items() if k != "filter_id"} 
            self.layer_info[l]["layer_size"]=lsize
        if self.approach=="original":
            self.P=self.init_assignment(True)
        else:
            self.P=self.init_assignment(False)
        self.Y=self.init_assignment(True)
        #test_assignment(self.Y,"failed init")
        self.V=self.zero_assignment()
        self.layers=[]
        self.W={}
        for (name, W) in model.named_parameters():  ## initialize Z (for both weights and bias)
            if name in self.prune_ratios:
                self.layers.append(name)
                self.W[name]=W
        self.init_convergence()

        print("!!!! ADMM runs with: ", self.approach, self.penalty, self.sparsity_type)
    def update_partition(self,newP):
        #print("PPPPPPPPPPPPPPPPPPPPPPPP")
        #print(self.config['partition'].keys())
        for l in self.layer_names:
            #print("LLLLLLLLLLLLLLLLLLLLLLLL")
            #print(self.partition[l]['filter_id'])
            #print(self.config["new_P"][l])
            self.partition[l]['filter_id']=new_to_old_partition(newP[l])
            #print(self.partition[l]['filter_id'])
        #print(self.config['partition'].keys())



    def ppp(self):
        print("#### debugging! ")
        name = self.layers[3]
        print(self.P[name])
        print(self.Y[name])

    def getW(self,model):
        Weights={}
        for (name, W) in model.named_parameters():
            if name in admm.prune_ratios:
                Weights[name]=W
        return Weights

    def test_prunning(self,Ws,Ps,s="mmmmm"):
        test=False
        zeros=0
        total=0
        zerosp=0
        totalp=0
        print(s)
        for (name,W) in Ws.items():
            #simple
            E=self.getE(W)
            E[E!=0]=1
            (n1,n2)=E.shape
            total=n1*n2
            zeros=total-E.sum()
            print(name,total,zeros,zeros/total)
            #lin alg
            P=self.getPin(name,Ps)
            needs=E@P
            needs[needs!=0]=1
            (n,m)=needs.shape
            totalp=n*m
            zerosp=totalp-needs.sum()
            print(name,totalp,zerosp,zerosp/totalp)
            print(name,(totalp-n),(zerosp-n),(zerosp-n)/(totalp-n))
            print(self.prune_ratios[name])
    
    def return_assignment(self,hard=True):
        if self.approach=="original":
            return self.P
        elif self.approach=="relaxed":
            if hard:
                return self.Y          
            return self.P
    def init_convergence(self):
        self.prev_W=copy.deepcopy(self.W)
        self.prev_Z=copy.deepcopy(self.ADMM_Z)
        self.prev_Y=copy.deepcopy(self.Y)
        self.prev_P=copy.deepcopy(self.P)
    def agreement(self):
        agreement_W = sum(torch.norm(self.W[name] - self.ADMM_Z[name]) for name in self.layers)
        agreement_P = sum(torch.norm(self.P[name] - self.Y[name]) for name in self.layers)
        return agreement_W,agreement_P
        
    def convergence(self):
        convergence_W = sum(torch.norm(self.W[name] - self.prev_W[name]) for name in self.layers)
        convergence_Z = sum(torch.norm(self.ADMM_Z[name] - self.prev_Z[name]) for name in self.layers)
        convergence_P = sum(torch.norm(self.P[name] - self.prev_P[name]) for name in self.layers)
        convergence_Y = sum(torch.norm(self.Y[name] - self.prev_Y[name]) for name in self.layers)
        #print("!!!! Convergence values: ", convergence_W,convergence_Z,convergence_P, convergence_Y)
        return convergence_W,convergence_Z,convergence_P,convergence_Y

    def W_update_old(self):
        self.prev_W=copy.deepcopy(self.W)


    def init_assignment(self,hard=False):
        P={}
        for l in self.layer_names:
            if hard==False:
                if l!="inputs":
                    P[l]=torch.full((self.layer_info[l]["layer_size"],self.num), 1/self.num).to(device='cuda')
                else:
                    P[l]=translate_to_tensor(self.partition[l]['filter_id'])
            else:
                P[l]=translate_to_tensor(self.partition[l]['filter_id'])
        return P

    def zero_assignment(self):
        P={}
        for l in self.layer_names:
            P[l]=P[l]=torch.zeros((self.layer_info[l]["layer_size"],self.num)).to(device='cuda')
        return P

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

    def compute_admm_loss(self, model):
        admm_loss = {}
        for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
            if name not in self.prune_ratios:
                continue

            #print(type(W), type(self.ADMM_Z[name]))
            admm_loss[name] = 0.5 * self.rhos[name] * (torch.norm(W - self.ADMM_Z[name] + self.ADMM_U[name], p=2)**2)
            
            #print(name, admm_loss[name], ADMM.rhos[name])
            #print(name, torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2)**2)
            # admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)  # test if Z,U are net detached


        total_admm_loss = 0
        for k, v in admm_loss.items():
            total_admm_loss += v
        #print('admm_loss: ', mixed_loss)
        return total_admm_loss


    def Z_update(self,config_dict,
               model,
               writer=False,
               cross_x=4,
               cross_f=1):
        self.prev_Z=copy.deepcopy(self.ADMM_Z)
        #print("Z update")
        admm_epochs, sparsity_type = config_dict['admm_epochs'],self.sparsity_type 
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in self.prune_ratios:
                continue          
            if config_dict['multi_rho']:
                admm_multi_rho_scheduler(self,name) # call multi rho scheduler every admm update
            
            self.ADMM_Z[name]=torch.zeros(W.shape).to(self.device)

            self.ADMM_Z[name] += W.detach() + self.ADMM_U[name].detach()  # Z(k+1) = W(k+1)+U[k]

            self.ADMM_Z[name]=self.WP(self.ADMM_Z[name],name,self.return_assignment(hard=True), sparsity_type, cross_x, cross_f)  # equivalent to Euclidean Projection
            
            
    
    def U_update(self,config_dict,
               model,
               writer=False,
               cross_x=4,
               cross_f=1):
        #print("U update")
        admm_epochs, sparsity_type = config_dict['admm_epochs'], self.sparsity_type
 
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in self.prune_ratios:
                continue
            self.ADMM_U[name] = torch.zeros(W.shape).to(self.device)
            self.ADMM_U[name] += W.detach() - self.ADMM_Z[name].detach() + self.ADMM_U[name].detach()  # U(k+1) = W(k+1) - Z(k+1) +U(k)

    def getPin(self,name,Ps):
        parents = self.partition[name].get('parents', [])
        P=copy.deepcopy(Ps[parents[0]])
        for i in range(1,len(parents)):
            P+=Ps[parents[i]]
        P[P!=0]=1
        return P

    def WP(self,weight,name,Ps, sparsity_type, cross_x=4, cross_f=1):
        prune_ratio=self.prune_ratios[name]
        partition=self.partition
        weight = weight.detach()
        device = weight.device
        percent = prune_ratio * 100
        if len(weight.shape)==2:
            W= weight
        else:
            W= torch.norm(weight, dim=(2, 3))
        parents = partition[name].get('parents', [])
        P=copy.deepcopy(Ps[parents[0]])
        
        for parent in parents: 
            if parent not in partition:
                raise ValueError(f"Parent layer {parent} not found in partition dictionary.")
        for i in range(1,len(parents)):
            P+=Ps[parents[i]]
        P[P!=0]=1
        
        if (sparsity_type == "irregular") :
            weight = weight.cpu().detach().numpy()
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
            #return torch.from_numpy(above_threshold).to(device), torch.from_numpy(weight).to(device)
            return torch.from_numpy(weight).to(device)
        elif (sparsity_type == 'kernel'):
            weight = weight.cpu().detach().numpy()
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
            #return above_threshold, torch.from_numpy(weight).to(device)
            return torch.from_numpy(weight).to(device)
        elif (sparsity_type == "column"):
            weight = weight.cpu().detach().numpy()
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
            #return torch.from_numpy(
            #    expand_above_threshold).to(device), torch.from_numpy(weight).to(device)
            return torch.from_numpy(weight).to(device)
    
        elif (sparsity_type == "row"):
            weight = weight.cpu().detach().numpy()
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
            #return torch.from_numpy(
            #    expand_above_threshold).to(device), torch.from_numpy(weight).to(device)
            return torch.from_numpy(weight).to(device)

        elif sparsity_type == "partition_row" :
            E=W**2
            needs=torch.sqrt(E@P)
            #W[W!=0]=1
            needs=W@P
            Pout=Ps[name]

            
            #adjust semantics pr is ratio of pruned to total
            total=needs.shape[0]*needs.shape[1]
            aligned_keep=needs.shape[0]
            misaligned_total=total-aligned_keep

            misaligned_keep=(1-prune_ratio)*misaligned_total            
            apr=misaligned_keep/total
            
            c2=1-Pout
            res1=needs*c2
            
            t=torch.quantile(res1, 1-apr) 
            res1[res1>t]=1
            res1[res1!=1]=0
            res1=res1+Pout

            mask=res1@torch.transpose(P,0,1)
            mask[mask!=0]=1
            if len(weight.shape)==2:
                weight=mask*weight
            else:
                mask=mask.unsqueeze(2)
                mask=mask.unsqueeze(3)
                mask=mask.repeat((1,1,weight.shape[2],weight.shape[3]))
                weight=mask*weight
            return weight
    
    def comunication_penalty(self,hard=False,dif=False,P=None):
        if P==None:
            if self.approach=="original":
                P=self.P
            if self.approach=="relaxed":
                if hard==False:
                    P=self.P
                else:
                    P=self.Y
        if self.penalty=="full":
            comm=comunication_penalty1(self.model,self,self.config,P)
        if self.penalty=="aggregate_partition_rows":
            comm=comunication_penalty2(self.model,self,self.config,P,dif)
        return comm  
    
    def getE(self,W,eval=False):
        E=torch.norm(W, dim=(2, 3))
        if eval:
            E[E!=0]=1
        return E
    def update_assignment(self):

        self.prev_P=copy.deepcopy(self.P)
        P_costs=[]
        if self.approach=="original":
            #print("timing P")
            start=time.time()            
            if self.reassign:
                solve_original_assignment(self,dif=True) 
            #print(time.time()-start)
            P_costs.append(self.comunication_penalty().item()) # TT
        elif self.approach=="relaxed":
            #print("timing P")
            start=time.time()
            if self.reassign:
                solve_relaxed_assignment(self.model,self,self.config) 
                #print(time.time()-start)
                start=time.time()
                #print("timing Y")
                Y_update(self.model,self,self.config) 
                #print(time.time()-start)
                start=time.time()
                #print("timing V")
                V_update(self.model,self,self.config) 
                #print(time.time()-start)
                start=time.time()
            P_costs.append(self.comunication_penalty(hard=True).item()) #TT
        return P_costs

    def linear_cost_matrix(self,name,dif=False):
        if self.approach=="original":
            Ps=self.P
        parents = self.partition[name].get('parents', [])

        E=torch.norm(self.W[name], dim=(2, 3))
        
        if len(parents)==0:
            raise ValueError(f"no parents")
        P=copy.deepcopy(Ps[parents[0]])
        for i in range(1,len(parents)):
            P+=Ps[parents[i]]
        #print(E.device,P.device,self.C.device)
        if self.penalty=="full":
            needs=E@P
        elif self.penalty=="aggregate_partition_rows":
            needs=E@P
            #print("@@@@@@@ needs: ", needs)
            if dif==False:
                needs[needs!=0]=1
            else:
                t=1
                needs=torch.tanh(t*needs)
        Cl=needs @ self.C

        #print(E.shape,P.shape,self.C.shape,Cl.shape)
        return Cl
        #if cost type is 2
    def finaliseW(self):
        model=self.model
        self.Z_update(self.config, model)
        for (name, W) in model.named_parameters():
            if name not in self.prune_ratios:  # ignore layers that do not have rho
                continue
            W.data = self.ADMM_Z[name]


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
import sys
from ..utils.assignment import *

from ..core.admm import *


def Y_update(model,ADMM,configs):
    ADMM.prev_Y=copy.deepcopy(ADMM.Y)
    for (name, W) in model.named_parameters():
        if name in ADMM.prune_ratios:
            #print(name)
            C=-ADMM.P[name] - ADMM.V[name]
            (n,m)=C.shape
            Y=np.zeros((n,m))
            #print(C.shape)
            #print(ADMM.ADMM_Y.P[name].shape)
            C=C.cpu().detach().numpy()
            sol, total_cost = computeassignment_scipy(C, configs['partition'][name]['budget'])
            for (i,j,c) in sol:
                Y[i,j]=1
            #print(sol)
            #print(Y)
            #print("XXXXXXXXXXXXXXXXXXXXX")
            #print(ADMM.ADMM_Y.P[name])
            #print("---------------------")
            ADMM.Y[name]=torch.from_numpy(Y).float().to(ADMM.device)
            #print(ADMM.ADMM_Y.P[name])
            #sys.exit()
    #test_assignment(ADMM.Y,s="failed hard assignment")

def test_assignment(P,s="?"):
    test=False
    for (k,v) in P.items():
        (n,m)=v.shape
        for i in range(n):
            sum=0
            for j in range(m):
                sum+=v[i,j]
                if v[i,j]!=0 and v[i,j]!=1:
                    print(v[i,:],i)
                    test=True
            if sum!=1:
                print(v[i,:],i)
                test=True
    if test:
        print(s)
        print(k,v)
        sys.exit()
def getE(W):
    return torch.norm(W, dim=(2, 3))
def test_prunning(model,Ps,admm,s="mmmmm"):
    test=False
    
    print(s)
    for (name, W) in model.named_parameters():
        if name in admm.prune_ratios:
            #simple
            zeros=0
            total=0
            zerosp=0
            totalp=0
            E=admm.getE(W)
            E[E!=0]=1
            (n1,n2)=E.shape
            total=n1*n2
            zeros=total-E.sum()
            print("kernels",name,total,zeros,zeros/total)
            #lin alg
            P=admm.getPin(name,Ps)
            needs=E@P
            needs[needs!=0]=1
            (n,m)=needs.shape
            totalp=n*m
            zerosp=totalp-needs.sum()
            print("rows",name,totalp,zerosp,zerosp/totalp)
            print("off rows",name,(totalp-n),(zerosp-n),(zerosp-n)/(totalp-n))
            print(admm.prune_ratios[name])




def V_update(model,ADMM,configs):
    for (name, W) in model.named_parameters():
        if name in ADMM.prune_ratios:
            #print("XXXXXXXXXXXXXXXXXXXXX")
            #print(ADMM.ADMM_V.P[name])
            #print("---------------------")
            #print(ADMM.V[name].device,ADMM.Y[name].device,ADMM.P[name].device)
            ADMM.V[name] = ADMM.P[name].detach() - ADMM.Y[name].detach() + ADMM.V[name].detach()
            #print(ADMM.ADMM_V.P[name])
            #sys.exit()

def layer_penalty_com(layer_name,ADMM,P):
    #TT 
    n=ADMM.layer_info[layer_name]['layer_size']
    parents = ADMM.layer_info[layer_name].get('parents', [])
    if len(parents)==0:
        return 0
    np=ADMM.layer_info[parents[0]]['layer_size']
    comm_costs = torch.zeros((np,n)).float().to(ADMM.device)
    C=ADMM.C.to(ADMM.device)
    for parent in parents:
        #print(ra.P[parent].shape)
        #print(ra.C.shape)
        #print(torch.transpose(ra.P[layer_name],0,1).shape )
        #print(ADMM.P[parent].device,C.device,ADMM.P[layer_name].device)
        
        comm_costs+=P[parent] @ C @ torch.transpose(P[layer_name],0,1)   
    #print(comm_costs.shape)
    return torch.transpose(comm_costs,0,1).to(ADMM.device)

def layer_penalty_com2(ADMM,name,E,P,dif=False, ret=False):
    #TT 
    #n=ADMM.layer_info[name]['layer_size']
    parents = ADMM.layer_info[name].get('parents', [])
    if len(parents)==0:
        return 0
    #np=ADMM.layer_info[parents[0]]['layer_size']
    comm_costs = 0
    C=ADMM.C.to(ADMM.device)
    for parent in parents:
        #print("----- parent: ", parent)
        Pin=P[parent]
        Pout=P[name]
        needs=E@Pin
        if dif==False:
            needs[needs!=0]=1
        else:
            t=1
            needs=torch.tanh(t*needs)
        costs=Pout@C
        res2=needs*costs
        comm_costs+=torch.sum(res2)
        #print(ra.P[parent].shape)
        #print("@@@@@ res2.shape:", res2.shape)
        #print(torch.transpose(ra.P[name],0,1).shape )
        #print(ADMM.P[parent].device,C.device,ADMM.P[name].device)  
    #print("&&&&& comm_costs: ", comm_costs)
    #if ret and name==ADMM.layers[3]:
        #print("###### needs: ", needs)
    return comm_costs

def comunication_penalty1(model,ADMM,configs,P,eval=False):
    comm_loss=0
    Ws=ADMM.W
    if eval:
        Ws=ADMM.ADMM_Z
    for name in ADMM.layers:
        W=Ws[name]
        #TT
        #print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",name)
        #print(torch.abs(W).sum((2,3)) .shape,penalty_com(ra,name,device="cuda").shape)
        E=ADMM.getE(W,eval)
        if eval:
            E[E!=0]=1
        comm_cost = E * layer_penalty_com(name,ADMM,P)
        comm_cost = comm_cost.view(comm_cost.size(0), -1).sum()
        #print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",name,comm_cost)
        if configs['comm_outsize']:
            comm_loss += comm_cost*ADMM.layer_info[name]['outsize']
        else:
            comm_loss += comm_cost
    return comm_loss

def comunication_penalty2(model,ADMM,configs,P,dif=False, ret=False,eval=False):
    comm_loss=0
    l = []
    ccms = []
    lys = []
    Ws=ADMM.W
    if eval:
        Ws=ADMM.ADMM_Z
    for name in ADMM.layers:
        W=Ws[name]
        #TT
        #print("+++++ name, W.shape:, ", name, W.shape)
        #print(torch.abs(W).sum((2,3)) .shape,penalty_com(ra,name,device="cuda").shape)
        E=ADMM.getE(W,eval)
        #print("+++++ E.shape:, ", name, E.shape)
        comm_cost = layer_penalty_com2(ADMM,name,E,P,dif, ret)
        ccms.append(comm_cost.item())
        #print("\n!!!!!!!!!!!!! after layer_penalty_com2 !!!!!!!!!!!!!!!!!!!!!!!!",name,comm_cost)
        if configs['comm_outsize']:
            comm_loss += comm_cost*ADMM.layer_info[name]['outsize']
            lys.append(ADMM.layer_info[name]['outsize'])
            l.append(comm_loss.item())

            #print("\n!!!!!!!!!!! ADMM.layer_info[name]['outsize']", ADMM.layer_info[name]['outsize'])
            #print("\n!!!!!!!!!!!!! in comm_outsize",comm_loss)
        else:
            comm_loss += comm_cost
            l.append(comm_loss.item())
            #print("\n!!!!!!!!!!!!! out comm_outsize",comm_loss)
        '''
        if ret and name==ADMM.layers[3]:
            print("###### P: ", P[name])
            print("###### W: ", W)
            print("###### E: ", E)
            print("###### comm_cost: ", comm_cost)
        '''   
    if ret:
        #print("======= ccms: ", ccms)
        #print("======= lys: ", lys)
        return comm_loss, l
    
    return comm_loss

def center_norm(model,ADMM,configs):
    norm=0
    for (name, W) in model.named_parameters(): 
        if name in ADMM.prune_ratios:
            norm += 0.5 * ADMM.rhos[name] * (torch.norm(ADMM.P[name] - ADMM.Y[name] + ADMM.V[name], p=2)**2)
    return norm

def solve_original_assignment(ADMM,dif):
    for name in ADMM.layers:
        #print(name)
        C=ADMM.linear_cost_matrix(name,dif)
        (n,m)=C.shape
        P=np.zeros((n,m))
        #print(C.shape)
        #print(ADMM.ADMM_Y.P[name].shape)
        C=C.cpu().detach().numpy()
        sol, total_cost = computeassignment_scipy(C, ADMM.config['partition'][name]['budget'])
        for (i,j,c) in sol:
            P[i,j]=1
        #print(sol)
        #print(Y)
        #print("XXXXXXXXXXXXXXXXXXXXX")
        #print(ADMM.ADMM_Y.P[name])
        #print("---------------------")
        ADMM.P[name]=torch.from_numpy(P).float().to(ADMM.device)
        #print(ADMM.ADMM_Y.P[name])


def solve_relaxed_assignment(model,ADMM,configs):
    P=[]
    for name in ADMM.layer_names:
        if name!='inputs':
            ADMM.P[name].requires_grad=True
            P.append(ADMM.P[name])
    
    
    optimizer = torch.optim.Adam(P, lr=0.01)
    for i in range(100):
        optimizer.zero_grad()
        with torch.enable_grad():
            
            loss = ADMM.comunication_penalty(dif=True,P=ADMM.P)
            #abs(loss)
            loss+=center_norm(model,ADMM,configs)
        loss.backward()
        optimizer.step()

def evaluate_comm(model,configs,P):
    admm=ADMM(configs, model, rho=configs['rho'])
    comm_cost=comunication_penalty1(model,admm,configs,P,eval=True)
    comm_cost2 =comunication_penalty2(model,admm,configs,P,eval=True)
    return comm_cost,comm_cost2




               
#########################################################################
#########################################################################

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


def compute_admm_loss(ADMM, model):
    admm_loss = {}
    for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
        if name not in ADMM.prune_ratios:
            continue

        admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2)**2)
        #print(name, admm_loss[name], ADMM.rhos[name])
        #print(name, torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2)**2)
        # admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)  # test if Z,U are net detached

    total_admm_loss = 0
    for k, v in admm_loss.items():
        total_admm_loss += v
    #print('admm_loss: ', mixed_loss)
    return total_admm_loss