from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import argparse
import numpy as np
import dataloader_clothing1M as dataloader # add pin_memory=True
from sklearn.mixture import GaussianMixture
import time
from collections import OrderedDict
import math
import json
from PreResNet_c1m import *
import matplotlib.pyplot as plt
from scipy.stats import kde
import shutil

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--id', default='gmm')
parser.add_argument('--data_path', default='../../Clothing1M/data', type=str, help='path to dataset')
parser.add_argument('--seed', default=1002)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=1000, type=int)

parser.add_argument('--rampup_epoch', default=2, type=int) # "... ramp up eta (meta-learning rate) from 0 to 0.4 during the first 20 epochs"
parser.add_argument('--num_ssl', default=10, type=int, help='number of intentional perturbations for SSL')
parser.add_argument('--fast_lr', default=0.002, type=float, help='learning rate for the temporarily-updated model for model training')
parser.add_argument('--meta_lr', default=0.004, type=float, help='learning rate for the meta loss')
parser.add_argument('--kl_epoch', default=2, type=int) # Epoch to start to record KL-divergence
parser.add_argument('--num_rw', default=10, type=int, help='number of intentional perturbations for sample reweighting')
parser.add_argument('--n_rw_epoch', default=0, type=int, help='number of epochs for sample reweighting')
parser.add_argument('--lr_rw', default=0.02, type=float, help='learning rate for sample reweighting')
parser.add_argument('--fast_lr_rw', default=0.0001, type=float, help='learning rate for the temporarily-updated model for sample reweighting')
parser.add_argument('--T_rw', default=5.0, type=float, help='sharpening temperature for sample weights')
parser.add_argument('--diag_multi', default=200.0, type=float, help='diagonal loss multiplication for sample reweighting')
parser.add_argument('--offd_multi', default=1.0, type=float, help='off-diagonal loss multiplication for sample reweighting')
parser.add_argument('--inv_off_diag', action='store_true', help='use inverse of off-diagonal KL div. as the objective function')
parser.add_argument('--num_warmup', default=1, type=int) # DivideMix: 10 for CIFAR-10 and 30 for CIFAR-100
parser.add_argument('--cotrain', action='store_true', help='train two models if present')
parser.add_argument('--r_penalty', default=1.0, type=float, help='weight of the confidence penalty for asymmetric noise')
parser.add_argument('--sw_fname1', default='sw1.npy', type=str, help='name of the saved sample weights in the range (-inf, inf)')
parser.add_argument('--sw_fname2', default='sw2.npy', type=str, help='name of the saved sample weights in the range (-inf, inf)')
parser.add_argument('--ckpt1', default='ckpt1.pth.tar', type=str, help='path to the 1st saved checkpoint of the DivideMix method')
parser.add_argument('--ckpt2', default='ckpt2.pth.tar', type=str, help='path to the 2nd saved checkpoint of the DivideMix method')
parser.add_argument('--best_acc1', default=0.0, type=float, help='best validation accuracy by the 1st saved checkpoint of the DivideMix method')
parser.add_argument('--best_acc2', default=0.0, type=float, help='best validation accuracy by the 2nd saved checkpoint of the DivideMix method')
parser.add_argument('--n_epoch_per_rw', default=1, type=int, help='number of E-step epochs between two consecutive M-steps')
parser.add_argument('--rw_start_epoch', default=60, type=int, help='number of epochs to start M-step')
parser.add_argument('--lr_decay_epoch', default=40, type=int, help='number of epochs to decrease lr')
parser.add_argument('--p_threshold_rw', default=0.5, type=float, help='clean probability threshold for prob_rw')

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
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # sys.stdout.write('\r')
        # sys.stdout.write('Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f '
        #         %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        # sys.stdout.flush()
        if batch_idx+1 == num_iter:
            print('Clothing1M (train) | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.4f'
                %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))

def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, _, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 

        # sys.stdout.write('\r')
        # sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
        #         %(batch_idx+1, args.num_batches, loss.item(), penalty.item()))
        # sys.stdout.flush()

def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    print("\n| Validation\t Net%d  Acc: %.2f%%" %(k,acc))  
    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
        print('| Saving Best Net%d ...'%k)
        #save_point = './checkpoint/%s_net%d.pth.tar'%(args.id,k)
        #torch.save(net.state_dict(), save_point)
        save_checkpoint({
            'state_dict': net.state_dict(),
        }, log_name+'_net%d.pth.tar'%k)
    return acc

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
    else:
        acc = 100.*correct/total
        print(prefix + " Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))

def reweighting(mentor_net, rw_loader, sample_weights, train_ep = 0, fname_number=1):
    mentor_net.train()
    #sample_weights = torch.zeros(50000, requires_grad=True, device='cuda')
    optimizer_rw = optim.SGD([sample_weights], lr=args.lr_rw, momentum=0.9, weight_decay=1e-4)
    for ep in range(args.n_rw_epoch):
        print('\n[REWEIGHTING] epoch %03d' % ep)
        start = time.time()
        bn_state = mentor_net.save_BN_state_dict()
        paths = []
        for batch_idx, (inputs, targets, sample_idx, path) in enumerate(rw_loader):
            if batch_idx == 0:
                print('in reweighting() epoch %d batch_idx 0, see first 5 paths:' % ep)
                for ii in range(5):
                    print(path[ii])
            inputs = inputs.cuda()
            targets = targets.cuda()
            sample_idx = sample_idx.cuda()
            for b in range(inputs.size(0)):
                paths.append(path[b])
            if batch_idx == 0:
                print(targets)
                print(sample_idx)
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
        # with torch.no_grad():
        #     prob = torch.sigmoid(sample_weights * args.T_rw).detach().cpu().numpy()
        #     auc_meter = AUCMeter()
        #     auc_meter.reset()
        #     auc_meter.add(prob,clean)        
        #     auc,_,_ = auc_meter.value()
        #     print('[reweighting] AUC%d: %.3f' % (fname_number, auc))
        #     #print('[reweighting] AUC:%.3f' % roc_auc_score(clean, sample_weights.cpu().numpy()))
        #     # plot density
        #     w_temp_clean = prob[clean]
        #     w_temp_noisy = prob[~clean]
        #     print('average weights for clean samples:', np.mean(w_temp_clean))
        #     print('average weights for noisy samples:', np.mean(w_temp_noisy))
        #     x = np.linspace(0,1,100)
        #     density_clean = kde.gaussian_kde(w_temp_clean)
        #     y_clean = density_clean(x)
        #     density_noisy = kde.gaussian_kde(w_temp_noisy)
        #     y_noisy = density_noisy(x)
        #     plt.plot(x, y_clean)
        #     plt.plot(x, y_noisy)
        #     plt.title('density%d_ep%03dep%02d.png (AUC: %.3f)' % (fname_number, train_ep, ep, auc))
        #     plt.savefig(os.path.join(density_path, 'density%d_ep%03dep%02d.png' % (fname_number, train_ep, ep)))
        #     plt.close()
        #     # inspect sample weights for bird
        #     prob_bird = prob[idx_bird]
        #     clean_bird = clean[idx_bird]
        #     auc_meter.reset()
        #     auc_meter.add(prob_bird,clean_bird)        
        #     auc,_,_ = auc_meter.value()
        #     print('[reweighting] AUC%d for bird: %.3f' % (fname_number, auc))
        #     w_temp_clean = prob_bird[clean_bird]
        #     w_temp_noisy = prob_bird[~clean_bird]
        #     print('average weights for clean bird samples:', np.mean(w_temp_clean))
        #     print('average weights for noisy bird samples:', np.mean(w_temp_noisy))
        #     x = np.linspace(0,1,100)
        #     density_clean = kde.gaussian_kde(w_temp_clean)
        #     y_clean = density_clean(x)
        #     density_noisy = kde.gaussian_kde(w_temp_noisy)
        #     y_noisy = density_noisy(x)
        #     plt.plot(x, y_clean)
        #     plt.plot(x, y_noisy)
        #     plt.title('density%d_bird_ep%03dep%02d.png (AUC: %.3f)' % (fname_number, train_ep, ep, auc))
        #     plt.savefig(os.path.join(density_path, 'density%d_bird_ep%03dep%02d.png' % (fname_number, train_ep, ep)))
        #     plt.close()
        #     # inspect sample weights for cat
        #     prob_cat = prob[idx_cat]
        #     clean_cat = clean[idx_cat]
        #     auc_meter.reset()
        #     auc_meter.add(prob_cat,clean_cat)        
        #     auc,_,_ = auc_meter.value()
        #     print('[reweighting] AUC%d for cat: %.3f' % (fname_number, auc))
        #     w_temp_clean = prob_cat[clean_cat]
        #     w_temp_noisy = prob_cat[~clean_cat]
        #     print('average weights for clean cat samples:', np.mean(w_temp_clean))
        #     print('average weights for noisy cat samples:', np.mean(w_temp_noisy))
        #     x = np.linspace(0,1,100)
        #     density_clean = kde.gaussian_kde(w_temp_clean)
        #     y_clean = density_clean(x)
        #     density_noisy = kde.gaussian_kde(w_temp_noisy)
        #     y_noisy = density_noisy(x)
        #     plt.plot(x, y_clean)
        #     plt.plot(x, y_noisy)
        #     plt.title('density%d_cat_ep%03dep%02d.png (AUC: %.3f)' % (fname_number, train_ep, ep, auc))
        #     plt.savefig(os.path.join(density_path, 'density%d_cat_ep%03dep%02d.png' % (fname_number, train_ep, ep)))
        #     plt.close()
        print('time elapsed:', time.time() - start)
    np.save(log_name + '_sw%d.npy' % fname_number, sample_weights.detach().cpu().numpy())
    return sample_weights, paths

def eval_train_acc(epoch,eval_loader,model,prefix='| Train'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, sample_index, path) in enumerate(eval_loader):
            if batch_idx == 0:
                print('in eval_train_acc() batch_idx 0, see first 5 paths:')
                for ii in range(5):
                    print(path[ii])
            inputs, targets = inputs.cuda(), targets.cuda() 
            total += targets.size(0)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)            
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print(prefix + " Epoch #%d\t Accuracy: %.2f%%" %(epoch,acc))

def eval_train(epoch,eval_loader,model):
    model.eval()
    num_samples = args.num_batches*args.batch_size
    losses = torch.zeros(num_samples)
    paths = []
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, sample_index, path) in enumerate(eval_loader):
            if batch_idx == 0:
                print('in eval_train() batch_idx 0, see first 5 paths:')
                for ii in range(5):
                    print(path[ii])
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[n]=loss[b] 
                paths.append(path[b])
                n+=1
            #sys.stdout.write('\r')
            #sys.stdout.write('| Evaluating loss Iter %3d\t' %(batch_idx)) 
            #sys.stdout.flush()
            
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    losses = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
    prob = prob[:,gmm.means_.argmin()]       
    return prob,paths  
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
               
pretrained_model = models.resnet50(pretrained=True)
def create_model():
    model = ResNet50(args.num_class)
    model.fc = nn.Linear(2048,args.num_class)
    for name, param in pretrained_model.state_dict().items():
        if not 'fc' in name:
            model.state_dict()[name].copy_(param)
    model = model.cuda()
    return model     

# log=open('./checkpoint/%s.txt'%args.id,'w')     
# log.flush()
log_name = './checkpoint/%s' % args.id
log_name = log_name + '_tau%sand%s' % (args.p_threshold, args.p_threshold_rw)
log_name = log_name + '_w%dep%drw%dper%dstart%d' % (args.num_warmup, args.num_epochs, args.n_rw_epoch, args.n_epoch_per_rw, args.rw_start_epoch)
if not args.cotrain:
    log_name = log_name + '_single'

if args.n_rw_epoch > 0:
    log_name = log_name + '_n%dd%dand%dlr%sf%sT%d' % (args.num_rw, args.diag_multi, args.offd_multi, args.lr_rw, args.fast_lr_rw, args.T_rw)

density_path = log_name + '_density'
if not os.path.exists(density_path):
    os.mkdir(density_path)

loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=4,num_batches=args.num_batches)

all_pairs = []
for i in range(args.num_class):
    for j in range(args.num_class):
        if not i == j:
            all_pairs.append(np.array([i,j]))

print('| Building net')
net1 = create_model()
# mentor_net1 = create_model()
if args.cotrain:
    net2 = create_model()
    mentor_net2 = create_model()
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
if args.cotrain:
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
consistent_criterion = nn.KLDivLoss(reduction='none')
softmax_dim1 = nn.Softmax(dim=1)
nll_loss = nn.NLLLoss()

lr=args.lr
test_loader = loader.run('test')
for param_group in optimizer1.param_groups:
    param_group['lr'] = lr       
if args.cotrain:
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

# load pre-trained model if specified; Otherwise, student <- teacher
ckpt_path_1 = os.path.join('checkpoint', args.ckpt1)
ckpt_path_2 = os.path.join('checkpoint', args.ckpt2)
if os.path.exists(ckpt_path_1):
    print('load net1 from %s' % ckpt_path_1)
    loaded_checkpoint = torch.load(ckpt_path_1)
    net1.load_state_dict(loaded_checkpoint['state_dict'])
    # mentor_net1.load_state_dict(loaded_checkpoint['state_dict'])
    if 'epoch' in loaded_checkpoint.keys():
        n_ep_ckpt1 = loaded_checkpoint['epoch']
    if 'optimizer_state_dict' in loaded_checkpoint.keys():
        optimizer1.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
if os.path.exists(ckpt_path_2):
    print('load net2 from %s' % ckpt_path_2)
    loaded_checkpoint = torch.load(ckpt_path_2)
    net2.load_state_dict(loaded_checkpoint['state_dict'])
    # mentor_net2.load_state_dict(loaded_checkpoint['state_dict'])
    if 'epoch' in loaded_checkpoint.keys():
        n_ep_ckpt2 = loaded_checkpoint['epoch']
    if 'optimizer_state_dict' in loaded_checkpoint.keys():
        optimizer2.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

# [WARMUP]
best_acc = [args.best_acc1, args.best_acc2]
for epoch in range(args.num_warmup):   
    print('[WARMUP] epoch %03d, lr %.6f' % (epoch, lr))
    
    start_epoch = time.time()
    print('Warmup Net1')
    warmup_trainloader = loader.run('warmup')
    warmup(net1,optimizer1,warmup_trainloader)
    print('(the whole warmup) time elapsed:', time.time() - start_epoch)
    if args.cotrain:
        start_epoch = time.time()
        print('\nWarmup Net2')
        warmup_trainloader = loader.run('warmup')
        warmup(net2,optimizer2,warmup_trainloader)
        print('(the whole warmup) time elapsed:', time.time() - start_epoch)

    val_loader = loader.run('val') # validation
    if args.cotrain:
        #eval_loader1 = loader.run('eval_train')
        #eval_train_acc(epoch,eval_loader1,net1,prefix='| Train net1')
        #eval_loader2 = loader.run('eval_train')
        #eval_train_acc(epoch,eval_loader2,net2,prefix='| Train net2')
        acc1 = val(net1,val_loader,1)
        acc2 = val(net2,val_loader,2)
        print('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
    else:
        #eval_loader = loader.run('eval_train')
        #eval_train_acc(epoch,eval_loader,net1)
        acc1 = val(net1,val_loader,1)
        print('Validation Epoch:%d      Acc1:%.2f\n'%(epoch,acc1))

# mentor_net1.load_state_dict(net1.state_dict())
# if args.cotrain:
#     mentor_net2.load_state_dict(net2.state_dict())

# [TRAIN]
for epoch in range(args.num_epochs):
    lr = args.lr
    if epoch >= (args.lr_decay_epoch * 2):
        lr = args.lr/100
    elif epoch >= args.lr_decay_epoch:
        lr = args.lr/10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr     
    if args.cotrain:
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr    
    
    # (0) Before RW, save the best Divide-Mix models for later reuse
    if epoch == args.rw_start_epoch:
        saved_best_net1 = log_name + '_net1.pth.tar'
        renamed_best_net1 = log_name + '_net1_backup.pth.tar'
        saved_best_net2 = log_name + '_net2.pth.tar'
        renamed_best_net2 = log_name + '_net2_backup.pth.tar'
        if os.path.exists(saved_best_net1):
            print('copy %s as %s' % (saved_best_net1, renamed_best_net1))
            shutil.copyfile(saved_best_net1, renamed_best_net1)
        if os.path.exists(saved_best_net2):
            print('copy %s as %s' % (saved_best_net2, renamed_best_net2))
            shutil.copyfile(saved_best_net2, renamed_best_net2)
    
    print('[TRAIN] epoch %03d, lr %.6f' % (epoch, lr))
    # (1) evaluate training data
    start_epoch = time.time()
    print('\n==== net 1 evaluate next epoch training data loss ====') 
    eval_loader1 = loader.run('eval_train')
    prob1,paths1 = eval_train(epoch,eval_loader1,net1)
    print('(the whole eval_train) time elapsed:', time.time() - start_epoch)
    
    # (2) M-step on net2 (note that 'rw_start_epoch % n_epoch_per_rw' must be 0, such that sw2 will be initialized)
    if epoch >= args.rw_start_epoch and args.n_rw_epoch > 0:
        if epoch % args.n_epoch_per_rw == 0:
            if epoch == args.rw_start_epoch and os.path.exists(os.path.join('checkpoint', args.sw_fname2)):
                start_epoch = time.time()
                print('load sample weights from checkpoint/%s' % args.sw_fname2)
                sw2 = torch.from_numpy(np.load(os.path.join('checkpoint', args.sw_fname2))).cuda()
                sw2.requires_grad = True
                prob_rw = torch.sigmoid(sw2 * args.T_rw).detach().cpu().numpy()
                print('(the whole reweighting for prob_rw) time elapsed:', time.time() - start_epoch)
            else:
                # [2022/09/28] find a bug: we should always initialize sw2!
                if True or epoch == args.rw_start_epoch:
                    print('initialize sw2')
                    sw2 = torch.zeros(args.num_batches*args.batch_size, requires_grad=True, device='cuda')
                start_epoch = time.time()
                if os.path.exists(saved_best_net2):
                    print('load model from %s' % saved_best_net2)
                    loaded_checkpoint = torch.load(saved_best_net2)
                    mentor_net2.load_state_dict(loaded_checkpoint['state_dict'])
                else:
                    print('load model from current net2')
                    mentor_net2.load_state_dict(net2.state_dict())
                mentor_net2.cuda()
                mentor_net2.train()
                test(-1,mentor_net2,'| (before RW) Test',None)  
                eval_loader2 = loader.run('eval_train')
                sw2, paths2 = reweighting(mentor_net2,eval_loader2,sw2,epoch,2)
                with torch.no_grad():
                    prob_rw = torch.sigmoid(sw2 * args.T_rw).detach().cpu().numpy()
                test(-1,mentor_net2,'| (after RW) Test',None)  
                print('(the whole reweighting for prob_rw) time elapsed:', time.time() - start_epoch)
            print('[paths1:]')
            print('len(paths1):', len(paths1))
            print('len(paths2):', len(paths2))
            for ii in range(10):
                print(paths1[ii])
                print(paths2[ii])
                print()
            # fit a two-component GMM to prob_rw
            prob_rw = prob_rw.reshape(-1,1)
            gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
            gmm.fit(prob_rw)
            prob2 = gmm.predict_proba(prob_rw) 
            prob2 = prob2[:,gmm.means_.argmax()]
    else:
        start_epoch = time.time()
        print('\n==== net 2 evaluate next epoch training data loss ====') 
        eval_loader2 = loader.run('eval_train')
        prob2,paths2 = eval_train(epoch,eval_loader2,net2)
        print('(the whole eval_train) time elapsed:', time.time() - start_epoch)
    pred1 = (prob1 > args.p_threshold)
    pred2 = (prob2 > args.p_threshold_rw)
    
    # plot density
    x1 = np.linspace(0,1,100)
    density1 = kde.gaussian_kde(prob1)
    y1 = density1(x1)
    x2 = np.linspace(0,1,100)
    density2 = kde.gaussian_kde(prob2)
    y2 = density2(x2)
    fig, ax = plt.subplots()
    ax.plot(x1, y1, label='prob1')
    ax.plot(x2, y2, label='prob2')
    ax.legend()
    fig.savefig(os.path.join(density_path, 'all_density_ep%03d.png' % epoch))
    plt.close(fig)
    
    # (3) E-step
    # Net1
    print('\n\nTrain Net1')
    labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2,paths=paths2) # co-divide
    start_epoch = time.time()
    train(epoch,net1,optimizer1,labeled_trainloader, unlabeled_trainloader,net2)              # train net1
    print('(the whole train) time elapsed:', time.time() - start_epoch)
    # Net2
    print('\nTrain Net2')
    labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1,paths=paths1) # co-divide
    start_epoch = time.time()
    train(epoch,net2,optimizer2,labeled_trainloader, unlabeled_trainloader,net1)              # train net2
    print('(the whole train) time elapsed:', time.time() - start_epoch)

    val_loader = loader.run('val') # validation
    if args.cotrain:
        #eval_train_acc(epoch,eval_loader1,net1,prefix='| Train net1')
        #eval_train_acc(epoch,eval_loader2,net2,prefix='| Train net2')
        acc1 = val(net1,val_loader,1)
        acc2 = val(net2,val_loader,2)
        print('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
    else:
        #eval_train_acc(epoch,eval_loader1,net1)
        acc1 = val(net1,val_loader,1)
        print('Validation Epoch:%d      Acc1:%.2f\n'%(epoch,acc1))
    # log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
    # log.flush()

    if os.path.exists(log_name+'_net1.pth.tar'):
        latest_model_name1 = log_name+'_net1.pth.tar'
    else:
        latest_model_name1 = ckpt_path_1
    if args.cotrain:
        if os.path.exists(log_name+'_net2.pth.tar'):
            latest_model_name2 = log_name+'_net2.pth.tar'
        else:
            latest_model_name2 = ckpt_path_2
    # if (epoch+1) % args.n_epoch_per_rw == 0:
    #     latest_model_name1 = log_name+'_net1_ep%d.pth.tar' % (n_ep_warmup_net1+epoch+1)
    #     save_checkpoint({
    #         'epoch': n_ep_warmup_net1+epoch+1,
    #         'state_dict': net1.state_dict(),
    #         'optimizer_state_dict': optimizer1.state_dict(),
    #     }, latest_model_name1)
    #     if args.cotrain:
    #         latest_model_name2 = log_name+'_net2_ep%d.pth.tar' % (n_ep_warmup_net2+epoch+1)
    #         save_checkpoint({
    #             'epoch': n_ep_warmup_net2+epoch+1,
    #             'state_dict': net2.state_dict(),
    #             'optimizer_state_dict': optimizer2.state_dict(),
    #         }, latest_model_name2)

print('\n[TEST]\n')
print('load model from %s' % latest_model_name1)
loaded_checkpoint = torch.load(latest_model_name1)
net1.load_state_dict(loaded_checkpoint['state_dict'])
print('load model from %s' % latest_model_name2)
loaded_checkpoint = torch.load(latest_model_name2)
net2.load_state_dict(loaded_checkpoint['state_dict'])
acc = test(-1,net1,'| Test',net2)
