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
np.set_printoptions(precision=4, suppress=True)

from PreResNet_slssl import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar_slssl as dataloader
import time
from collections import OrderedDict
import math

import json
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

from torchnet.meter import AUCMeter

import matplotlib.pyplot as plt
from scipy.stats import kde

#from sklearn.metrics import roc_auc_score

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

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
parser.add_argument('--id', default='em')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='/data', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)

parser.add_argument('--rampup_epoch', default=10, type=int) # "... ramp up eta (meta-learning rate) from 0 to 0.4 during the first 20 epochs"
parser.add_argument('--num_ssl', default=10, type=int, help='number of intentional perturbations for SSL')
parser.add_argument('--fast_lr', default=0.02, type=float, help='learning rate for the temporarily-updated model for model training')
parser.add_argument('--meta_lr', default=0.04, type=float, help='learning rate for the meta loss')
parser.add_argument('--kl_epoch', default=10, type=int) # Epoch to start to record KL-divergence
parser.add_argument('--num_rw', default=10, type=int, help='number of intentional perturbations for sample reweighting')
parser.add_argument('--n_rw_epoch', default=0, type=int, help='number of epochs for sample reweighting')
parser.add_argument('--lr_rw', default=0.2, type=float, help='learning rate for sample reweighting')
parser.add_argument('--fast_lr_rw', default=0.001, type=float, help='learning rate for the temporarily-updated model for sample reweighting')
parser.add_argument('--T_rw', default=5.0, type=float, help='sharpening temperature for sample weights')
parser.add_argument('--diag_multi', default=200.0, type=float, help='diagonal loss multiplication for sample reweighting')
parser.add_argument('--offd_multi', default=10.0, type=float, help='off-diagonal loss multiplication for sample reweighting')
parser.add_argument('--inv_off_diag', action='store_true', help='use inverse of off-diagonal KL div. as the objective function')
parser.add_argument('--prob_combine_r', default=0.5, type=float, help='ratio to combine original clean probability and the sample weights')
parser.add_argument('--num_warmup', default=10, type=int) # DivideMix: 10 for CIFAR-10 and 30 for CIFAR-100
parser.add_argument('--cotrain', action='store_true', help='train two models if present')
parser.add_argument('--tch_model_1', default='tch1.pth.tar', type=str, help='name of the saved teacher model')
parser.add_argument('--tch_model_2', default='tch2.pth.tar', type=str, help='name of the saved teacher model')
parser.add_argument('--r_penalty', default=1.0, type=float, help='weight of the confidence penalty for asymmetric noise')
parser.add_argument('--sw_fname1', default='sw1.npy', type=str, help='name of the saved sample weights in the range (-inf, inf)')
parser.add_argument('--sw_fname2', default='sw2.npy', type=str, help='name of the saved sample weights in the range (-inf, inf)')
parser.add_argument('--n_epoch_per_rw', default=10, type=int, help='number of E-step epochs between two consecutive M-steps')
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# Training
def train(epoch,net,optimizer,labeled_trainloader,unlabeled_trainloader,net2=None):
    net.train()
    if not net2 is None:
        net2.eval() #fix one network and train the other
    
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
            if not net2 is None:
                outputs_u11 = net(inputs_u)
                outputs_u12 = net(inputs_u2)
                outputs_u21 = net2(inputs_u)
                outputs_u22 = net2(inputs_u2)            
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            else:
                outputs_u11 = net(inputs_u)
                outputs_u12 = net(inputs_u2)
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2 
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
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, args.num_warmup)
        #Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, args.num_warmup, tm_tensor)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #sys.stdout.write('\r')
        #sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
        #        %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        #sys.stdout.flush()
        if batch_idx+1 == num_iter:
            print('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))

def warmup(epoch,net,tch_net,optimizer,dataloader,tm,init):
    #start = time.time()
    net.train()
    tch_net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    
    init_return = init
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
    
    #print('(pre) time elapsed:', time.time() - start)
    for batch_idx, (inputs, targets, index) in enumerate(dataloader):      
        inputs, targets = inputs.cuda(), targets.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs_prob = softmax_dim1(outputs)
        outputs_adjust = torch.mm(outputs_prob, tm_tensor)
        outputs_log_prob = torch.log(outputs_adjust)
        #nll_loss_all = nll_loss(outputs_log_prob, targets)
        #loss = torch.matmul(nll_loss_all, loss_weights) / len(targets)
        loss = nll_loss(outputs_log_prob, targets)
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + args.r_penalty * penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()

        if args.num_ssl > 0 and epoch >= 2:
            if init_return:
                init_return = False
                #print('initialize tch_net using net')
                for param, param_tch in zip(net.parameters(),tch_net.parameters()):
                    param_tch.data.copy_(param.data)
            else:
                #print('update tch_net by EMA over net')
                for param, param_tch in zip(net.parameters(),tch_net.parameters()):
                    param_tch.data.mul_(gamma).add_((1-gamma), param.data)
            tch_outputs = tch_net(inputs)
            p_tch = F.softmax(tch_outputs, dim=1)
            p_tch = p_tch.detach()
            
            for i in range(args.num_ssl):
                #start = time.time()
                targets_fast = targets.clone()
                rand_lb_pair = np.random.choice(range(args.num_class), size=2, replace=True) # note: we allow C1 == C2 here
                idx0 = [idx for idx in range(targets.size(0)) if targets[idx] == rand_lb_pair[0]]
                for n in range(targets.size(0)):
                    if n in idx0:
                        targets_fast[n] = rand_lb_pair[1]
                loss_mask = torch.cuda.FloatTensor(args.num_class).fill_(1.0)
                for idx in rand_lb_pair:
                    loss_mask[idx] = 0.0
                
                # Forward again
                outputs = net(inputs)
                fast_loss = CEloss(outputs, targets_fast)
                
                # Single-step gradient descent
                grads = torch.autograd.grad(fast_loss, net.parameters())
                for grad in grads:
                    grad = grad.detach()
                    grad.requires_grad = False
                fast_weights = OrderedDict((name, param - args.fast_lr*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))
                
                # Compute the KL divergence
                fast_out = net.forward(inputs, fast_weights, is_training=True)
                logp_fast = F.log_softmax(fast_out, dim=1)
                kl_div_vector = consistent_criterion(logp_fast, p_tch)
                kl_div_reduced = torch.mean(kl_div_vector)
                kl_div_masked = torch.matmul(kl_div_vector, loss_mask) / args.num_class
                
                # Collect KL statistics
                rand_lb_pair_tuple = (rand_lb_pair[0], rand_lb_pair[1])
                if epoch >= args.kl_epoch:
                    if rand_lb_pair_tuple in kl_dict.keys():
                        kl_dict[rand_lb_pair_tuple].append(kl_div_reduced.data.item())
                    else:
                        kl_dict[rand_lb_pair_tuple] = [kl_div_reduced.data.item()]

                # Update the model using meta-loss
                meta_loss = meta_lr * torch.mean(kl_div_masked, dim=0) / args.num_ssl
                meta_loss.backward()
                #print('[batch %03d i %d] time elapsed:' % (batch_idx, i), time.time() - start)
        optimizer.step() 

        #sys.stdout.write('\r')
        #sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
        #        %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        #sys.stdout.flush()
        if batch_idx+1 == num_iter:
            print('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))

    # update tm
    #start = time.time()
    if args.num_ssl > 0 and epoch >= args.kl_epoch:
        tm_from_kl = np.zeros([args.num_class, args.num_class])
        for i in range(args.num_class):
            for j in range(args.num_class):
                if (i, j) in kl_dict.keys():
                    tm_from_kl[i][j] = np.mean(kl_dict[(i,j)])
        tm_from_kl = np.power(tm_from_kl+1e-10, -sharpen)
        row_sums = tm_from_kl.sum(axis=1, keepdims=True)
        tm_from_kl = tm_from_kl / row_sums
        tm = tm_keep_r * tm + (1 - tm_keep_r) * tm_from_kl
    #print('(post) time elapsed:', time.time() - start)
    return tm, init_return

def test(epoch,net1,prefix='| Test',net2=None):
    net1.eval()
    if not net2 is None:
        net2.eval()
    correct1 = 0
    correct2 = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            total += targets.size(0)
            if not net2 is None:
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)           
                outputs = outputs1+outputs2
                _, predicted1 = torch.max(outputs1, 1)            
                _, predicted2 = torch.max(outputs2, 1)            
                _, predicted = torch.max(outputs, 1)            
                correct1 += predicted1.eq(targets).cpu().sum().item()                 
                correct2 += predicted2.eq(targets).cpu().sum().item()                 
                correct += predicted.eq(targets).cpu().sum().item()                 
            else:
                outputs = net1(inputs)
                _, predicted = torch.max(outputs, 1)            
                correct += predicted.eq(targets).cpu().sum().item()                 
                       
    if not net2 is None:
        acc1 = 100.*correct1/total
        acc2 = 100.*correct2/total
        acc = 100.*correct/total
        print(prefix + " Epoch #%d\t Accuracy1: %.2f%%" %(epoch,acc1))  
        print(prefix + " Epoch #%d\t Accuracy2: %.2f%%" %(epoch,acc2))  
        print(prefix + " Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
        #log.write(prefix + ' Epoch:%d   Accuracy1:%.2f\n'%(epoch,acc1))
        #log.write(prefix + ' Epoch:%d   Accuracy2:%.2f\n'%(epoch,acc2))
        #log.write(prefix + ' Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    else:
        acc = 100.*correct/total
        print(prefix + " Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
        #log.write(prefix + ' Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))

def reweighting(mentor_net, dataloader, sample_weights, train_ep = 0, fname_number=1):
    mentor_net.train()
    #sample_weights = torch.zeros(50000, requires_grad=True, device='cuda')
    optimizer_rw = optim.SGD([sample_weights], lr=args.lr_rw, momentum=0.9, weight_decay=1e-4)
    for ep in range(args.n_rw_epoch):
        print('\n[REWEIGHTING] epoch %03d' % ep)
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
            
            loss_weights = torch.sigmoid(sample_weights.index_select(dim=0, index=sample_idx) * args.T_rw)
            targets_fast = targets.clone()
            fast_loss = torch.matmul(CE(mentor_outputs, targets_fast), loss_weights)
            grads = torch.autograd.grad(fast_loss, mentor_net.parameters(), create_graph=True)
            fast_weights = OrderedDict((name, param - args.fast_lr_rw*grad) for ((name, param), grad) in zip(mentor_net.named_parameters(), grads))
            fast_out = mentor_net.forward(x=inputs, weights=fast_weights, is_training=False)
                
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
            random.shuffle(all_pairs)
            for i in range(args.num_rw):
                targets_fast = targets.clone()
                # choose C1 and C2 for all mini-batches in this epoch
                #rand_lb_pair = np.random.choice(range(args.num_class), size=2, replace=False) # note: we want C1 != C2 here
                rand_lb_pair = all_pairs[i]
                idx0 = [idx for idx in range(targets.size(0)) if targets[idx] == rand_lb_pair[0]]
                if batch_idx == 0:
                    print('i={}, (C1, C2)={}, len(idx0)={}'.format(i, rand_lb_pair, len(idx0)))
                for n in range(targets.size(0)):
                    if n in idx0:
                        targets_fast[n] = rand_lb_pair[1]
                # forward again
                mentor_outputs = mentor_net(inputs)
                #sample_idx_C1C2 = [sample_idx[idx] for idx in range(targets.size(0)) if targets[idx] == rand_lb_pair[0] or targets[idx] == rand_lb_pair[1]]
                #sample_idx_C1C2 = torch.tensor(sample_idx_C1C2).cuda()
                #if batch_idx == 0:
                #    print('i={}, sample_idx.size(): {}, sample_idx_C1C2.size(); {}'.format(i, sample_idx.size(), sample_idx_C1C2.size()))
                loss_weights = torch.sigmoid(sample_weights.index_select(dim=0, index=sample_idx) * args.T_rw)
                fast_loss = torch.matmul(CE(mentor_outputs,targets_fast), loss_weights)
                grads = torch.autograd.grad(fast_loss, mentor_net.parameters(), create_graph=True)
                fast_weights = OrderedDict((name, param - args.fast_lr_rw*grad) for ((name, param), grad) in zip(mentor_net.named_parameters(), grads))
                fast_out = mentor_net.forward(x=inputs, weights=fast_weights, is_training=False)
                logp_fast = F.log_softmax(fast_out,dim=1)
                kl_div_vector = consistent_criterion(logp_fast, p_tch)
                kl_div_reduced = torch.mean(kl_div_vector)*args.offd_multi
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
                if args.num_rw > 0:
                    print('rw_loss_off_diag:', rw_loss_off_diag)
                    with torch.no_grad():
                        print('rw_loss_off_diag/rw_loss_diagonal=%.4f' % torch.true_divide(rw_loss_off_diag, rw_loss_diagonal).data.item())
                        print('rw_loss_diagonal/rw_loss_off_diag**2=%.4f' % torch.true_divide(rw_loss_diagonal, rw_loss_off_diag**2).data.item())
            optimizer_rw.step()
        # compute AUC
        with torch.no_grad():
            prob = torch.sigmoid(sample_weights * args.T_rw).detach().cpu().numpy()
            auc_meter = AUCMeter()
            auc_meter.reset()
            auc_meter.add(prob,clean)        
            auc,_,_ = auc_meter.value()
            print('[reweighting] AUC%d: %.3f' % (fname_number, auc))
            #print('[reweighting] AUC:%.3f' % roc_auc_score(clean, sample_weights.cpu().numpy()))
            # plot density
            w_temp_clean = prob[clean]
            w_temp_noisy = prob[~clean]
            print('average weights for clean samples:', np.mean(w_temp_clean))
            print('average weights for noisy samples:', np.mean(w_temp_noisy))
            x = np.linspace(0,1,100)
            density_clean = kde.gaussian_kde(w_temp_clean)
            y_clean = density_clean(x)
            density_noisy = kde.gaussian_kde(w_temp_noisy)
            y_noisy = density_noisy(x)
            plt.plot(x, y_clean)
            plt.plot(x, y_noisy)
            plt.title('density%d_ep%03dep%02d.png (AUC: %.3f)' % (fname_number, train_ep, ep, auc))
            plt.savefig(os.path.join(density_path, 'density%d_ep%03dep%02d.png' % (fname_number, train_ep, ep)))
            plt.close()
            # inspect sample weights for bird
            prob_bird = prob[idx_bird]
            clean_bird = clean[idx_bird]
            auc_meter.reset()
            auc_meter.add(prob_bird,clean_bird)        
            auc,_,_ = auc_meter.value()
            print('[reweighting] AUC%d for bird: %.3f' % (fname_number, auc))
            w_temp_clean = prob_bird[clean_bird]
            w_temp_noisy = prob_bird[~clean_bird]
            print('average weights for clean bird samples:', np.mean(w_temp_clean))
            print('average weights for noisy bird samples:', np.mean(w_temp_noisy))
            x = np.linspace(0,1,100)
            density_clean = kde.gaussian_kde(w_temp_clean)
            y_clean = density_clean(x)
            density_noisy = kde.gaussian_kde(w_temp_noisy)
            y_noisy = density_noisy(x)
            plt.plot(x, y_clean)
            plt.plot(x, y_noisy)
            plt.title('density%d_bird_ep%03dep%02d.png (AUC: %.3f)' % (fname_number, train_ep, ep, auc))
            plt.savefig(os.path.join(density_path, 'density%d_bird_ep%03dep%02d.png' % (fname_number, train_ep, ep)))
            plt.close()
            # inspect sample weights for cat
            prob_cat = prob[idx_cat]
            clean_cat = clean[idx_cat]
            auc_meter.reset()
            auc_meter.add(prob_cat,clean_cat)        
            auc,_,_ = auc_meter.value()
            print('[reweighting] AUC%d for cat: %.3f' % (fname_number, auc))
            w_temp_clean = prob_cat[clean_cat]
            w_temp_noisy = prob_cat[~clean_cat]
            print('average weights for clean cat samples:', np.mean(w_temp_clean))
            print('average weights for noisy cat samples:', np.mean(w_temp_noisy))
            x = np.linspace(0,1,100)
            density_clean = kde.gaussian_kde(w_temp_clean)
            y_clean = density_clean(x)
            density_noisy = kde.gaussian_kde(w_temp_noisy)
            y_noisy = density_noisy(x)
            plt.plot(x, y_clean)
            plt.plot(x, y_noisy)
            plt.title('density%d_cat_ep%03dep%02d.png (AUC: %.3f)' % (fname_number, train_ep, ep, auc))
            plt.savefig(os.path.join(density_path, 'density%d_cat_ep%03dep%02d.png' % (fname_number, train_ep, ep)))
            plt.close()
        print('time elapsed:', time.time() - start)
    np.save(log_name + '_sw%d.npy' % fname_number, sample_weights.detach().cpu().numpy())
    return sample_weights

def eval_train_acc(epoch,net1,net2=None,prefix='| Train'):
    net1.eval()
    if not net2 is None:
        net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            total += targets.size(0)
            if net2 is None:
                outputs = net1(inputs)
            else:
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)            
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print(prefix + " Epoch #%d\t Accuracy: %.2f%%" %(epoch,acc))

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

def linear_rampup(current, num_warmup, rampup_length=16):
    current = np.clip((current-num_warmup) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, num_warmup):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,num_warmup)
#class SemiLoss(object):
#    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, num_warmup, tm_tensor):
#        outputs_u_prob = torch.softmax(outputs_u, dim=1)
#        probs_u = torch.mm(outputs_u_prob, tm_tensor)
#        Lu = torch.mean((probs_u - targets_u)**2)
#        
#        outputs_x_prob = torch.softmax(outputs_x, dim=1)
#        probs_x = torch.mm(outputs_x_prob, tm_tensor)
#        log_probs_x = torch.log(probs_x)
#        Lx = -torch.mean(torch.sum(log_probs_x * targets_x, dim=1))
#        
#        return Lx, Lu, linear_rampup(epoch,num_warmup)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

log_name = './checkpoint/%s_%s%s'%(args.id, args.noise_mode, args.r)
log_name = log_name + '_lr%sf%s' % (args.lr, args.fast_lr)
if not args.lambda_u == 25:
    log_name = log_name + '_u%d' % args.lambda_u
if not args.p_threshold == 0.5:
    log_name = log_name + '_tau%s' % args.p_threshold
if not args.batch_size == 64:
    log_name = log_name + '_bs%d' % args.batch_size
if not args.r_penalty == 1.0:
    log_name = log_name + '_pen%s' % args.r_penalty
log_name = log_name + '_w%dep%drw%dper%d' % (args.num_warmup, args.num_epochs, args.n_rw_epoch, args.n_epoch_per_rw)
if not args.cotrain:
    log_name = log_name + '_single'
if args.n_rw_epoch > 0:
    log_name = log_name + '_n%dd%dand%dlr%sf%s' % (args.num_rw, args.diag_multi, args.offd_multi, args.lr_rw, args.fast_lr_rw)
    if not args.T_rw == 5.0:
        log_name = log_name + 'T%d' % args.T_rw
    if not args.prob_combine_r == 0.5:
        log_name = log_name + 'comb%s' % args.prob_combine_r
    density_path = log_name + '_density'
    if not os.path.exists(density_path):
        os.mkdir(density_path)
#stats_log=open(log_name+'_stats.txt','w') 
#test_log=open(log_name+'_acc.txt','w')     
#test_tch_log=open(log_name+'_acc_tch.txt','w')     

#if args.dataset=='cifar10':
#    num_warmup = 10
#elif args.dataset=='cifar100':
#    num_warmup = 30

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=4,\
    root_dir=args.data_path,log='',noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

noise_label = json.load(open('%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode),"r"))
train_label=[]
if args.dataset=='cifar10': 
    for n in range(1,6):
        dpath = '%s/data_batch_%d'%(args.data_path,n)
        data_dic = unpickle(dpath)
        train_label = train_label+data_dic['labels']
elif args.dataset=='cifar100':    
    train_dic = unpickle('%s/train'%root_dir)
    train_label = train_dic['fine_labels']
clean = (np.array(noise_label)==np.array(train_label))                                                       
idx_bird = [i for i in range(len(train_label)) if train_label[i] == 2]
idx_cat = [i for i in range(len(train_label)) if train_label[i] == 3]

all_pairs = []
for i in range(args.num_class):
    for j in range(args.num_class):
        if not i == j:
            all_pairs.append(np.array([i,j]))

print('| Building net')
net1 = create_model()
tch_net1 = create_model()
mentor_net1 = create_model()
if args.cotrain:
    net2 = create_model()
    tch_net2 = create_model()
    mentor_net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
if args.cotrain:
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
for param in tch_net1.parameters():
    param.requires_grad = False
if args.cotrain:
    for param in tch_net2.parameters():
        param.requires_grad = False

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()
consistent_criterion = nn.KLDivLoss(reduction='none')
softmax_dim1 = nn.Softmax(dim=1)
#nll_loss = nn.NLLLoss(reduction='none')
nll_loss = nn.NLLLoss()

all_loss = [[],[]] # save the history of losses from two networks
tm = [np.eye(args.num_class), np.eye(args.num_class)]
tm_keep_r = 0.99
sharpen = 20.0
tch_init = [True, True]
lr=args.lr
warmup_trainloader = loader.run('warmup')
test_loader = loader.run('test')
eval_loader = loader.run('eval_train')   
for param_group in optimizer1.param_groups:
    param_group['lr'] = lr       
if args.cotrain:
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          

# [WARMUP]
for epoch in range(args.num_warmup):   
    print('[WARMUP] epoch %03d, lr %.6f' % (epoch, lr))
    
    start_epoch = time.time()
    print('Warmup Net1')
    tm[0], tch_init[0] = warmup(epoch,net1,tch_net1,optimizer1,warmup_trainloader,tm[0], tch_init[0])    
    print('(the whole warmup) time elapsed:', time.time() - start_epoch)
    if args.cotrain:
        start_epoch = time.time()
        print('Warmup Net2')
        tm[1], tch_init[1] = warmup(epoch,net2,tch_net2,optimizer2,warmup_trainloader,tm[1], tch_init[1]) 
        print('(the whole warmup) time elapsed:', time.time() - start_epoch)
    
    if args.cotrain:
        eval_train_acc(epoch,net1,net2)
        test(epoch,net1,'| Test',net2)  
        test(epoch,tch_net1,'| tch Test',tch_net2)  
    else:
        eval_train_acc(epoch,net1)
        test(epoch,net1,'| Test',None)  
        test(epoch,tch_net1,'| tch Test',None)  

# save model at the last epoch
'''
https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
When saving a general checkpoint, you must save more than just the model’s state_dict. It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are updated as the model trains.
'''
if args.num_warmup > 0:
    save_checkpoint({
        'epoch': args.num_warmup,
        'state_dict': net1.state_dict(),
        'optimizer_state_dict': optimizer1.state_dict(),
    }, log_name+'_stu1.pth.tar')
    save_checkpoint({
        'epoch': args.num_warmup,
        'state_dict': tch_net1.state_dict(),
        'optimizer_state_dict': optimizer1.state_dict(),
    }, log_name+'_tch1.pth.tar')
    if args.cotrain:
        save_checkpoint({
            'epoch': args.num_warmup,
            'state_dict': net2.state_dict(),
            'optimizer_state_dict': optimizer2.state_dict(),
        }, log_name+'_stu2.pth.tar')
        save_checkpoint({
            'epoch': args.num_warmup,
            'state_dict': tch_net2.state_dict(),
            'optimizer_state_dict': optimizer2.state_dict(),
        }, log_name+'_tch2.pth.tar')

# load pre-trained model if specified; Otherwise, student <- teacher
tch_path_1 = os.path.join('checkpoint', args.tch_model_1)
tch_path_2 = os.path.join('checkpoint', args.tch_model_2)
if os.path.exists(tch_path_1):
    print('load model from %s' % tch_path_1)
    loaded_checkpoint = torch.load(tch_path_1)
    n_ep_warmup_net1 = loaded_checkpoint['epoch']
    net1.load_state_dict(loaded_checkpoint['state_dict'])
    mentor_net1.load_state_dict(loaded_checkpoint['state_dict'])
    if 'optimizer_state_dict' in loaded_checkpoint.keys():
        optimizer1.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    if args.cotrain and os.path.exists(tch_path_2):
        print('load model from %s' % tch_path_2)
        loaded_checkpoint = torch.load(tch_path_2)
        n_ep_warmup_net2 = loaded_checkpoint['epoch']
        net2.load_state_dict(loaded_checkpoint['state_dict'])
        mentor_net2.load_state_dict(loaded_checkpoint['state_dict'])
        if 'optimizer_state_dict' in loaded_checkpoint.keys():
            optimizer2.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
elif args.num_warmup > 0:
    print('load model from %s' % log_name+'_tch1.pth.tar')
    loaded_checkpoint = torch.load(log_name+'_tch1.pth.tar')
    n_ep_warmup_net1 = loaded_checkpoint['epoch']
    net1.load_state_dict(loaded_checkpoint['state_dict'])
    mentor_net1.load_state_dict(loaded_checkpoint['state_dict'])
    if 'optimizer_state_dict' in loaded_checkpoint.keys():
        optimizer1.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    if args.cotrain:
        print('load model from %s' % log_name+'_tch2.pth.tar')
        loaded_checkpoint = torch.load(log_name+'_tch2.pth.tar')
        n_ep_warmup_net2 = loaded_checkpoint['epoch']
        net2.load_state_dict(loaded_checkpoint['state_dict'])
        mentor_net2.load_state_dict(loaded_checkpoint['state_dict'])
        if 'optimizer_state_dict' in loaded_checkpoint.keys():
            optimizer2.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

# check if the student models are correctly set
if args.cotrain:
    test(-1,net1,'| (debug) Test',net2)
else:
    test(-1,net1,'| (debug) Test',None)  

# [TRAIN]
#for param_group in optimizer1.param_groups:
#    param_group['lr'] = lr/10
#if args.cotrain:
#    for param_group in optimizer2.param_groups:
#        param_group['lr'] = lr/10

for epoch in range(args.num_epochs):
    #if epoch >= 100:
    #    lr = args.lr/10
    #else:
    #    lr = args.lr
    lr = args.lr/2**(epoch//args.n_epoch_per_rw)
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    if args.cotrain:
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr          
    print('[TRAIN] epoch %03d, lr %.6f' % (epoch, lr))
    # (1) evaluate training data
    start_epoch = time.time()
    prob1,all_loss[0]=eval_train(net1,all_loss[0])   
    print('(the whole eval_train) time elapsed:', time.time() - start_epoch)
    if args.cotrain:
        start_epoch = time.time()
        prob2,all_loss[1]=eval_train(net2,all_loss[1])          
        print('(the whole eval_train) time elapsed:', time.time() - start_epoch)
    # (2) M-step
    if args.n_rw_epoch > 0:
        if epoch % args.n_epoch_per_rw == 0:
            if epoch == 0 and os.path.exists(os.path.join('checkpoint', args.sw_fname1)):
                start_epoch = time.time()
                print('load sample weights from checkpoint/%s' % args.sw_fname1)
                sw1 = torch.from_numpy(np.load(os.path.join('checkpoint', args.sw_fname1))).cuda()
                sw1.requires_grad = True
                prob1_rw = torch.sigmoid(sw1 * args.T_rw).detach().cpu().numpy()
                print('(the whole reweighting for prob1_rw) time elapsed:', time.time() - start_epoch)
                if args.cotrain:
                    start_epoch = time.time()
                    print('load sample weights from checkpoint/%s' % args.sw_fname2)
                    sw2 = torch.from_numpy(np.load(os.path.join('checkpoint', args.sw_fname2))).cuda()
                    sw2.requires_grad = True
                    prob2_rw = torch.sigmoid(sw2 * args.T_rw).detach().cpu().numpy()
                    print('(the whole reweighting for prob2_rw) time elapsed:', time.time() - start_epoch)
            else:
                if epoch > 0:
                    # load latest model
                    print('load mentor_net1 from %s' % latest_model_name1)
                    loaded_checkpoint = torch.load(latest_model_name1)
                    mentor_net1.load_state_dict(loaded_checkpoint['state_dict'])
                else:
                    print('initialize sw1')
                    sw1 = torch.zeros(50000, requires_grad=True, device='cuda')
                    if args.cotrain:
                        print('initialize sw2')
                        sw2 = torch.zeros(50000, requires_grad=True, device='cuda')
                start_epoch = time.time()
                mentor_net1.cuda()
                mentor_net1.train()
                test(-1,mentor_net1,'| (before RW) Test',None)  
                sw1 = reweighting(mentor_net1, warmup_trainloader, sw1, epoch, 1)
                prob1_rw = torch.sigmoid(sw1 * args.T_rw).detach().cpu().numpy()
                test(-1,mentor_net1,'| (after RW) Test',None)  
                print('(the whole reweighting for prob1_rw) time elapsed:', time.time() - start_epoch)
                if args.cotrain:
                    if epoch > 0:
                        # load latest model
                        print('load mentor_net2 from %s' % latest_model_name2)
                        loaded_checkpoint = torch.load(latest_model_name2)
                        mentor_net2.load_state_dict(loaded_checkpoint['state_dict'])
                    start_epoch = time.time()
                    mentor_net2.cuda()
                    mentor_net2.train()
                    test(-1,mentor_net2,'| (before RW) Test',None)  
                    sw2 = reweighting(mentor_net2, warmup_trainloader, sw2, epoch, 2)
                    prob2_rw = torch.sigmoid(sw2 * args.T_rw).detach().cpu().numpy()
                    test(-1,mentor_net2,'| (after RW) Test',None)  
                    print('(the whole reweighting for prob2_rw) time elapsed:', time.time() - start_epoch)
        prob1 = args.prob_combine_r * prob1 + (1 - args.prob_combine_r) * prob1_rw
        if args.cotrain:
            prob2 = args.prob_combine_r * prob2 + (1 - args.prob_combine_r) * prob2_rw
    pred1 = (prob1 > args.p_threshold)      
    if args.cotrain:
        pred2 = (prob2 > args.p_threshold)      
    # (3) E-step
    if args.cotrain:
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        start_epoch = time.time()
        train(epoch,net1,optimizer1,labeled_trainloader, unlabeled_trainloader,net2) # train net1  
        print('(the whole train) time elapsed:', time.time() - start_epoch)
        
        print('Train Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        start_epoch = time.time()
        train(epoch,net2,optimizer2,labeled_trainloader, unlabeled_trainloader,net1) # train net2         
        print('(the whole train) time elapsed:', time.time() - start_epoch)
    else:
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        start_epoch = time.time()
        train(epoch,net1,optimizer1,labeled_trainloader, unlabeled_trainloader,None) # train net1  
        print('(the whole train) time elapsed:', time.time() - start_epoch)
    
    if args.cotrain:
        eval_train_acc(epoch,net1,net2)
        test(epoch,net1,'| Test',net2)  
    else:
        eval_train_acc(epoch,net1)
        test(epoch,net1,'| Test',None)  

    if (epoch+1) % args.n_epoch_per_rw == 0:
        latest_model_name1 = log_name+'_net1_ep%d.pth.tar' % (n_ep_warmup_net1+epoch+1)
        save_checkpoint({
            'epoch': n_ep_warmup_net1+epoch+1,
            'state_dict': net1.state_dict(),
            'optimizer_state_dict': optimizer1.state_dict(),
        }, latest_model_name1)
        if args.cotrain:
            latest_model_name2 = log_name+'_net2_ep%d.pth.tar' % (n_ep_warmup_net2+epoch+1)
            save_checkpoint({
                'epoch': n_ep_warmup_net2+epoch+1,
                'state_dict': net2.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
            }, latest_model_name2)

