from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
import time
from collections import OrderedDict
import math

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='slssl')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='/data', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)

parser.add_argument('--rampup_epoch', default=20, type=int) # "... ramp up eta (meta-learning rate) from 0 to 0.4 during the first 20 epochs"
parser.add_argument('--num_ssl', default=10, type=int, help='number of intentional perturbations for SSL')
parser.add_argument('--fast_lr', default=0.02, type=float, help='learning rate for the temporarily-updated model for model training')
parser.add_argument('--meta_lr', default=0.04, type=float, help='learning rate for the meta loss')
parser.add_argument('--kl_epoch', default=0, type=int) # Epoch to start to record KL-divergence
parser.add_argument('--num_rw', default=10, type=int, help='number of intentional perturbations for sample reweighting')
parser.add_argument('--n_rw_epoch', default=0, type=int, help='number of epochs for sample reweighting')
parser.add_argument('--lr_rw', default=0.02, type=float, help='learning rate for sample reweighting')
parser.add_argument('--fast_lr_rw', default=0.001, type=float, help='learning rate for the temporarily-updated model for sample reweighting')
parser.add_argument('--diag_multi', default=200.0, type=float, help='diagonal loss multiplication for sample reweighting')
parser.add_argument('--inv_off_diag', action='store_true', help='use inverse of off-diagonal KL div. as the objective function')
parser.add_argument('--prob_combine_r', default=0.5, type=float, help='ratio to combine original clean probability and the sampple weights')
parser.add_argument('--warm_up', default=10, type=int) # DivideMix: 10 for CIFAR-10 and 30 for CIFAR-100

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch,net,tch_net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,tm,init):
    start = time.time()
    net.train()
    tch_net.train()
    net2.eval() #fix one network and train the other
    
    init_return = True
    tm_tensor = torch.Tensor(tm)
    tm_tensor = tm_tensor.cuda()
    tm_tensor = tm_tensor.detach()
    print('tm_tensor:', tm_tensor)
    kl_dict = {}
    if epoch > args.rampup_epoch:
        meta_lr = args.meta_lr
        gamma = 0.999
    else:
        u = epoch/args.rampup_epoch
        meta_lr = args.meta_lr * math.exp(-5*(1-u)**2)
        gamma = 0.99
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x_raw, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x_raw.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, args.warm_up, tm_tensor)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.num_ssl > 0 and epoch >= args.warm_up:
            if init:
                init_return = False
                for param, param_tch in zip(net.parameters(),tch_net.parameters()):
                    param_tch.data.copy_(param.data)
            else:
                for param, param_tch in zip(net.parameters(),tch_net.parameters()):
                    param_tch.data.mul_(gamma).add_((1-gamma), param.data)
            tch_outputs_x = tch_net(inputs_x)
            tch_outputs_x2 = tch_net(inputs_x2)
            p_tch = F.softmax(tch_outputs_x + tch_outputs_x2, dim=1)
            p_tch = p_tch.detach()
            
            labels_x_raw = labels_x_raw.cuda()
            for i in range(args.num_ssl):
                targets_fast = labels_x_raw.clone()
                rand_lb_pair = np.random.choice(range(args.num_class), size=2, replace=True) # note: we allow C1 == C2 here
                idx0 = [idx for idx in range(labels_x_raw.size(0)) if labels_x_raw[idx] == rand_lb_pair[0]]
                for n in range(labels_x_raw.size(0)):
                    if n in idx0:
                        targets_fast[n] = rand_lb_pair[1]
                loss_mask = torch.cuda.FloatTensor(args.num_class).fill_(1.0)
                for idx in rand_lb_pair:
                    loss_mask[idx] = 0.0
                
                # Forward again
                outputs_x = net(inputs_x)
                outputs_x2 = net(inputs_x2)
                fast_loss = CEloss(outputs_x + outputs_x2, targets_fast)
                
                # Single-step gradient descent
                grads = torch.autograd.grad(fast_loss, net.parameters())
                for grad in grads:
                    grad = grad.detach()
                    grad.requires_grad = False
                fast_weights = OrderedDict((name, param - args.fast_lr*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))
                
                # Compute the KL divergence
                fast_out = net.forward(inputs_x, fast_weights)
                fast_out2 = net.forward(inputs_x2, fast_weights)
                logp_fast = F.log_softmax(fast_out + fast_out2, dim=1)
                kl_div_vector = consistent_criterion(logp_fast, p_tch)
                kl_div_reduced = torch.mean(kl_div_vector)
                kl_div_masked = torch.matmul(kl_div_vector, loss_mask) / args.num_class
                
                # Collect KL statistics
                rand_lb_pair_tuple = (rand_lb_pair[0], rand_lb_pair[1])
                if epoch >= args.kl_epoch + args.warm_up:
                    if rand_lb_pair_tuple in kl_dict.keys():
                        kl_dict[rand_lb_pair_tuple].append(kl_div_reduced.data.item())
                    else:
                        kl_dict[rand_lb_pair_tuple] = [kl_div_reduced.data.item()]

                # Update the model using meta-loss
                meta_loss = meta_lr * torch.mean(kl_div_masked, dim=0) / args.num_ssl
                meta_loss.backward()
        
        optimizer.step()
        #sys.stdout.write('\r')
        #sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
        #        %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        #sys.stdout.flush()
        if batch_idx+1 == num_iter:
            print('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
    
    # update tm
    if args.num_ssl > 0 and epoch >= args.kl_epoch + args.warm_up:
        tm_from_kl = np.zeros([args.num_class, args.num_class])
        for i in range(args.num_class):
            for j in range(args.num_class):
                if (i, j) in kl_dict.keys():
                    tm_from_kl[i][j] = np.mean(kl_dict[(i,j)])
        tm_from_kl = np.power(tm_from_kl+1e-10, -sharpen)
        row_sums = tm_from_kl.sum(axis=1, keepdims=True)
        tm_from_kl = tm_from_kl / row_sums
        tm = tm_keep_r * tm + (1 - tm_keep_r) * tm_from_kl
    print('time elapsed:', time.time() - start)
    return tm, init_return

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 

        #sys.stdout.write('\r')
        #sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
        #        %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        #sys.stdout.flush()
        if batch_idx+1 == num_iter:
            print('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))

def test(epoch,net1,net2,log,prefix='| Test'):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print(prefix + " Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    log.write(prefix + ' Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    #test_log.flush()  

def reweighting(mentor_net, dataloader):
    sample_weights = torch.zeros(50000, requires_grad=True, device='cuda')
    optimizer_rw = optim.SGD([sample_weights], lr=args.lr_rw, momentum=0.9, weight_decay=1e-4)
    for ep in range(args.n_rw_epoch):
        print('\n=> Reweighting Epoch #%d' % ep)
        start = time.time()
        bn_state = mentor_net.save_BN_state_dict()
        for batch_idx, (inputs, targets, sample_idx) in enumerate(dataloader):
            inputs, targets, sample_idx = inputs.cuda(), targets.cuda(), sample_idx.cuda()
            optimizer_rw.zero_grad()
            mentor_net.eval()
            mentor_outputs_base = mentor_net(inputs) # BN in evaluation mode: use running statistics and do not update them
            p_tch = F.softmax(mentor_outputs_base, dim=1)
            p_tch = p_tch.detach()
            mentor_net.train()
            mentor_outputs = mentor_net(inputs) # BN in training mode: use batch statistics and update running statistics

            loss_weights = torch.sigmoid(sample_weights.index_select(dim=0, index=sample_idx) / args.T)
            targets_fast = targets.clone()
            fast_loss = torch.matmul(CE(mentor_outputs, targets_fast), loss_weights)
            grads = torch.autograd.grad(fast_loss, mentor_net.parameters(), create_graph=True)
            fast_weights = OrderedDict((name, param - args.fast_lr_rw*grad) for ((name, param), grad) in zip(mentor_net.named_parameters(), grads))
            fast_out = mentor_net.forward(x=inputs, weights=fast_weights)
                
            logp_fast = F.log_softmax(fast_out,dim=1)
            kl_div_vector = consistent_criterion(logp_fast, p_tch)
            rw_loss_diagonal = torch.mean(kl_div_vector)
            if batch_idx == 0:
                print('rw_loss_diagonal (before amplify):', rw_loss_diagonal)
            rw_loss_diagonal *= args.diag_multi
            rw_loss_diagonal.backward()
            mentor_net.load_BN_state_dict(bn_state)
            if batch_idx == 0:
                print('rw_loss_diagonal (after amplify):', rw_loss_diagonal)

            # C1 --> C2, 1-step GD
            for i in range(args.num_rw):
                targets_fast = targets.clone()
                # choose C1 and C2 for all mini-batches in this epoch
                rand_lb_pair = np.random.choice(range(args.num_class), size=2, replace=False) # note: we want C1 != C2 here
                idx0 = [idx for idx in range(targets.size(0)) if targets[idx] == rand_lb_pair[0]]
                for n in range(targets.size(0)):
                    if n in idx0:
                        targets_fast[n] = rand_lb_pair[1]
                # forward again
                mentor_outputs = mentor_net(inputs)
                loss_weights = torch.sigmoid(sample_weights.index_select(dim=0, index=sample_idx) / args.T)
                fast_loss = torch.matmul(CE(mentor_outputs,targets_fast), loss_weights)
                grads = torch.autograd.grad(fast_loss, mentor_net.parameters(), create_graph=True)
                fast_weights = OrderedDict((name, param - args.fast_lr_rw*grad) for ((name, param), grad) in zip(mentor_net.named_parameters(), grads))
                fast_out = mentor_net.forward(x=inputs, weights=fast_weights)
                logp_fast = F.log_softmax(fast_out,dim=1)
                kl_div_vector = consistent_criterion(logp_fast, p_tch)
                kl_div_reduced = torch.mean(kl_div_vector)
                if args.inv_off_diag:
                    rw_loss_others = 1/kl_div_reduced
                else:
                    rw_loss_others = -kl_div_reduced
                if batch_idx == 0:
                    with torch.no_grad():
                        if i == 0:
                            rw_loss_off_diag = kl_div_reduced
                        else:
                            rw_loss_off_diag += kl_div_reduced
                rw_loss_others.backward()
                mentor_net.load_BN_state_dict(bn_state)
            if batch_idx == 0:
                print('rw_loss_off_diag:', rw_loss_off_diag)
                with torch.no_grad():
                    print('rw_loss_off_diag/rw_loss_diagonal=%.4f' % torch.true_divide(rw_loss_off_diag, rw_loss_diagonal).data.item())
                    print('rw_loss_diagonal/rw_loss_off_diag**2=%.4f' % torch.true_divide(rw_loss_diagonal, rw_loss_off_diag**2).data.item())
            optimizer_rw.step()
        print('time elapsed:', time.time() - start)
    return torch.sigmoid(sample_weights / args.T).detach().cpu().numpy()

def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(50000)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up, tm_tensor):
        # probs_u = torch.softmax(outputs_u, dim=1)
        outputs_u_prob = torch.softmax(outputs_u, dim=1)
        probs_u = torch.mm(outputs_u_prob, tm_tensor)
        Lu = torch.mean((probs_u - targets_u)**2)
        
        # Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        outputs_x_prob = torch.softmax(outputs_x, dim=1)
        probs_x = torch.mm(outputs_x_prob, tm_tensor)
        log_probs_x = torch.log(probs_x)
        Lx = -torch.mean(torch.sum(log_probs_x * targets_x, dim=1))
        
        # Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

log_name = './checkpoint/%s_%s_%.1f_%s'%(args.id, args.dataset,args.r,args.noise_mode)
if not args.lambda_u == 25:
    log_name = log_name + '_u%d' % args.lambda_u
if not args.p_threshold == 0.5:
    log_name = log_name + '_tau%f' % args.p_threshold
if not args.batch_size == 64:
    log_name = log_name + '_bs%d' % args.batch_size
if args.n_rw_epoch > 0:
    log_name = log_name + '_rw'
log_name = log_name + '_ep%d' % args.num_epochs
stats_log=open(log_name+'_stats.txt','w') 
test_log=open(log_name+'_acc.txt','w')     
test_tch_log=open(log_name+'_acc_tch.txt','w')     

#if args.dataset=='cifar10':
#    warm_up = 10
#elif args.dataset=='cifar100':
#    warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=0,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net1 = create_model()
tch_net1 = create_model()
net2 = create_model()
tch_net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
for param in tch_net1.parameters():
    param.requires_grad = False
for param in tch_net2.parameters():
    param.requires_grad = False

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()
consistent_criterion = nn.KLDivLoss(reduction='none')

all_loss = [[],[]] # save the history of losses from two networks
tm = [np.eye(args.num_class), np.eye(args.num_class)]
tm_keep_r = 0.99
sharpen = 20.0
tch_init = [True, True]
for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<args.warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:         
        prob1,all_loss[0]=eval_train(net1,all_loss[0])   
        prob2,all_loss[1]=eval_train(net2,all_loss[1])          
        if args.n_rw_epoch > 0:
            prob1_rw = reweighting(net2, warmup_trainloader)
            prob2_rw = reweighting(net1, warmup_trainloader)
            prob1 = args.prob_combine_r * prob1 + (1 - args.prob_combine_r) * prob1_rw
            prob2 = args.prob_combine_r * prob2 + (1 - args.prob_combine_r) * prob2_rw
               
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        tm[0], tch_init[0] = train(epoch,net1,tch_net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader,tm[0], tch_init[0]) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        tm[1], tch_init[1] = train(epoch,net2,tch_net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader,tm[1], tch_init[1]) # train net2         

    test(epoch,net1,net2,test_log,prefix='| Test')  
    if args.num_ssl > 0:
        test(epoch,tch_net1,tch_net2,test_tch_log,prefix='| tch Test')  


