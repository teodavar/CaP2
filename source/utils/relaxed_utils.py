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
            '''
            for i in range(n1):
                for j in range(n2):
                    total+=1
                    if E[i,j]==0:
                        zeros+=1
            '''
            total=n1*n2
            zeros=total-E.sum()
            print(name,total,zeros,zeros/total)
            #lin alg
            P=admm.getPin(name,Ps)
            needs=E@P
            needs[needs!=0]=1
            (n,m)=needs.shape
            '''
            for i in range(n):
                for j in range(m):
                    totalp+=1
                    if needs[i,j]==0:
                        zerosp+=1
            '''
            totalp=n*m
            zerosp=totalp-needs.sum()
            print(name,totalp,zerosp,zerosp/totalp)
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
        E=ADMM.getE(W)
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
        E=ADMM.getE(W)
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

def evaluate_comm(model,admm,configs,P):
    comm_cost=comunication_penalty1(model,admm,configs,P,eval=True)
    comm_cost2 =comunication_penalty2(model,admm,configs,P,eval=True)
    return comm_cost,comm_cost2