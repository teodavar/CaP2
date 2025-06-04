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


def Y_update(model,ADMM,configs):
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

def layer_penalty_com2(ADMM,name,E,P,dif=False):
    #TT 
    n=ADMM.layer_info[name]['layer_size']
    parents = ADMM.layer_info[name].get('parents', [])
    if len(parents)==0:
        return 0
    np=ADMM.layer_info[parents[0]]['layer_size']
    comm_costs = 0
    C=ADMM.C.to(ADMM.device)
    for parent in parents:
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
        #print(ra.C.shape)
        #print(torch.transpose(ra.P[name],0,1).shape )
        #print(ADMM.P[parent].device,C.device,ADMM.P[name].device)  
    #print(comm_costs.shape)
    return comm_costs

def comunication_penalty(model,ADMM,configs,P):
    comm_loss=0
    for (name, W) in model.named_parameters():
        if name in ADMM.prune_ratios:
            #TT
            #print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",name)
            #print(torch.abs(W).sum((2,3)) .shape,penalty_com(ra,name,device="cuda").shape)
            E=ADMM.getE(W)
            comm_cost = E * layer_penalty_com(name,ADMM,P)
            comm_cost = comm_cost.view(comm_cost.size(0), -1).sum()
            #print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",name,comm_cost)
            if configs['comm_outsize']:
                comm_loss += comm_cost*ADMM.layer_info[name]['outsize']
            else:
                comm_loss += comm_cost
    return comm_loss

def comunication_penalty2(model,ADMM,configs,P,dif=False):
    comm_loss=0
    for (name, W) in model.named_parameters():
        if name in ADMM.prune_ratios:
            #TT
            #print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",name)
            #print(torch.abs(W).sum((2,3)) .shape,penalty_com(ra,name,device="cuda").shape)
            E=ADMM.getE(W)
            comm_cost = layer_penalty_com2(ADMM,name,E,P,dif)
            #print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",name,comm_cost)
            if configs['comm_outsize']:
                comm_loss += comm_cost*ADMM.layer_info[name]['outsize']
            else:
                comm_loss += comm_cost
    return comm_loss

def center_norm(model,ADMM,configs):
    norm=0
    for (name, W) in model.named_parameters(): 
        if name in ADMM.prune_ratios:
            norm += 0.5 * ADMM.rhos[name] * (torch.norm(ADMM.P[name] - ADMM.Y[name] + ADMM.V[name], p=2)**2)
    return norm

def solve_original_assignment(ADMM):
    for name in ADMM.layers:
        #print(name)
        C=ADMM.linear_cost_matrix(name)
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

