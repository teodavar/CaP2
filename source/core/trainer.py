import time
from tqdm import tqdm
from ..utils.misc import *
from ..utils.eval import *
from ..utils.relaxed_utils import *
from ..utils.assignment import *
from .admm import *
import sys

def standard_train(configs, cepoch, model, data_loader, criterion, optimizer, scheduler, ADMM=None, masks=None, comm=False, old_comm_loss=False):

    batch_acc    = AverageMeter()
    batch_loss   = AverageMeter()
    batch_total_loss   = AverageMeter()
    batch_comm   = AverageMeter()
    P_costs = []
    evalHelper   = EvalHelper(configs['data_code'])
    
    if not model.training:
        model.train()
    
    if comm:
        partition = configs['partition']
    
    if ADMM is not None: 
        ADMM.Z_update(configs, model)
        ADMM.W_update_old()

    
    start_time = time.time()
    n_data = configs['batch_size'] * len(data_loader)
    pbar = tqdm(enumerate(data_loader), total=n_data/configs['batch_size'], ncols=150)
    
    for batch_idx, batch in pbar:
        #print("!!!!!!!!!!! BATCH IDX: ", batch_idx)
        # TT
        # Run for smaller dataset
        #'''
        if batch_idx > 20:
            break
        #'''
        data   = ()
        for piece in batch[:-1]:
            data += (piece.float().to(configs['device']),)
        target = batch[-1].to(configs['device'])
        total_loss = 0
        comm_loss = 0
        comp_loss = 0

        data = (torch.cat(data, dim=1),)
        
        optimizer.zero_grad()
        
        if configs['mix_up']:
            data, target_a, target_b, lam = mixup_data(*data, y=target, alpha=configs['alpha'])
        # print('data:', data)
        
        # print(len(data))
        # print(data[0].shape)
        output = model(*data)
        # print('output:', output)
        
        if configs['mix_up']:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam, configs['smooth'])
        else:
            loss = criterion(output, target, smooth=configs['smooth'])
            # loss = criterion(output, target.unsqueeze(1).float())
        # print('xentropy_loss:', loss)
        total_loss += (loss * configs['xentropy_weight'])
        # print('total_loss:', total_loss)
        
        if ADMM is not None:
            admm_loss= ADMM.compute_admm_loss(model)  
            total_loss+=admm_loss
            
        if ADMM is not None and comm: 
            comm_loss=ADMM.comunication_penalty(dif=True)
                    
            total_loss += configs['lambda_comm'] * comm_loss + configs['lambda_comp'] * comp_loss
            # print('total_loss:', total_loss)
        
        total_loss.backward() # Back Propagation
        
        # For masked training
        if masks is not None:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks and W.grad is not None:
                        W.grad *= masks[name]
                        
        optimizer.step()
        
        # adjust learning rate
        if ADMM is not None:
            admm_adjust_learning_rate(optimizer, cepoch, configs)
        else:
            scheduler.step()
        c=False  # TT
        
        if (cepoch != 1 and batch_idx == 0 and (cepoch-1) % configs['theta_epochs'] == 0) or c:
            if ADMM is not None:
                print("\n!!!!!! Running Updates for ", cepoch, batch_idx)
                ADMM.Z_update(configs, model)
                ADMM.U_update(configs, model)
                
            if ADMM is not None and configs['reassign']: #TT
                #print("#### Re assign")
                P_costs=ADMM.update_assignment()



        acc1 = evalHelper.call(output, target)
        batch_loss.update(loss.item(), target.size(0))
        batch_total_loss.update(total_loss.item(), target.size(0))
        batch_comm.update(comm_loss.item() if comm_loss else comm_loss, target.size(0))
        batch_acc.update(acc1[0].item(), target.size(0))

        
        # # # preparation log information and print progress # # #
        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] Loss:{loss:.4f} CommLoss:{commloss:.4f} Acc:{acc:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*configs['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*configs['batch_size']/n_data), 
                        loss = batch_loss.avg,
                        commloss = batch_comm.avg,
                        acc  = batch_acc.avg,
                    )

        pbar.set_description(msg)
    # TT
    # Debugging
    #ADMM.ppp()
    #print('Training time per epoch is {:.2f}s.'.format(time.time()-start_time))
    metrics = {
        'batch_loss': batch_loss,
        'batch_total_loss': batch_total_loss,
        'batch_comm': batch_comm,
        'batch_acc':  batch_acc,
        'admm_loss': admm_loss if ADMM is not None else None,
        'P_costs': P_costs
    }
    return metrics

    
def distill_train(configs, cepoch, teacher, student, data_loader, optimizer, scheduler):

    batch_acc    = AverageMeter()
    distillloss   = AverageMeter()
    filtloss   = AverageMeter()
    
    n_data = configs['batch_size'] * len(data_loader)
    
    start_time = time.time()
    pbar = tqdm(enumerate(data_loader), total=n_data/configs['batch_size'], ncols=150)
    for batch_idx, (data, target) in pbar:
           
        data   = data.to(configs['device'])
        target = target.to(configs['device'])
        
        optimizer.zero_grad()
        
        t_output, t_filt = teacher(data)
        s_output, s_filt = student(data)
        distill_loss = distillation(s_output, t_output, target, configs['distill_temp'], configs['distill_alpha'],)
        filt_loss = sum([actTransfer_loss(x, y) for x, y in zip([s_filt], [t_filt])])
        loss = distill_loss + configs['lambda_f']*filt_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc1 = accuracy(s_output, target, topk=(1,))
        distillloss.update(distill_loss.item(), data.size(0))
        filtloss.update(filt_loss.item(), data.size(0))
        batch_acc.update(acc1[0].item(), data.size(0))

        # # # preparation log information and print progress # # #
        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] DistillLoss:{distillloss:.4f} FiltLoss:{filtloss:.4f} Acc:{acc:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*configs['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*configs['batch_size']/n_data),
                        distillloss = distillloss.avg,
                        filtloss = filtloss.avg,
                        acc  = batch_acc.avg,
                    )

        pbar.set_description(msg)
    #print('Training time per epoch is {:.2f}s.'.format(time.time()-start_time))
    
