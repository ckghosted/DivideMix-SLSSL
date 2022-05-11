from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models # [2021/09/28] important

import dataloader_cifar_slssl as dataloader
import numpy as np
from sklearn.cluster import KMeans
import json, os, argparse, re
import pickle
from collections import Counter

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        # https://stackoverflow.com/questions/46001958/typeerror-a-bytes-like-object-is-required-not-str-when-opening-python-2-pick/47814305#47814305
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
parser.add_argument('--split_file', default='train_val_split.json', type=str, help='indices that specify training and validation split')
parser.add_argument('--feat_fname', default='resnet50_feat.pkl', type=str, help='file name of the extracted features')
parser.add_argument('--seed', default=1002, type=int, help='random seed for k-means clustering')
parser.add_argument('--plot_heatmap', action='store_true', help='if set, plot the heatmap of class transition matrix')
args = parser.parse_args()

resnet_depth = re.search('resnet([0-9]+)_feat.pkl', args.feat_fname).group(1)
noise_file = os.path.join(args.data_path, '0.0_rn{}.json'.format(resnet_depth))
if os.path.exists(noise_file):
    print('load existing noise_file: %s' % noise_file)
    noise_label = json.load(open(noise_file, 'r'))
    targets_train = json.load(open(os.path.join(args.data_path, 'gt_labels.json'), 'r'))
    
    ## compute the noise rate
    noise_count = 0
    for i in range(len(targets_train)):
        if not noise_label[i] == targets_train[i]:
            noise_count += 1
    print('noise rate: %.4f' % (noise_count / len(targets_train)))
else:
    print('generate pseudo labels by applying k-means++ to features extracted from an ImageNet-pretrained ResNet')

    ## load pre-trained model
    feat_path = os.path.join(args.data_path, args.feat_fname)
    if os.path.exists(feat_path):
        print('load the saved features')
        resnet_feat = unpickle(feat_path)
        features_all = resnet_feat['features']
        targets_all = resnet_feat['labels']
    else:
        print('extract features from an ImageNet-pretrained ResNet')
        ### load Imagenet-pretrained ResNet model
        resnet_feat = {}
        model_selector = {'18': models.resnet18, '50': models.resnet50}
        resnet = model_selector[resnet_depth](pretrained=True)
        ### Ref: https://stackoverflow.com/questions/62117707/extract-features-from-pretrained-resnet50-in-pytorch
        resnet.fc = nn.Identity()
        resnet = resnet.cuda()
        resnet.eval()
        ### create a clean data loader (without noise)
        loader = dataloader.cifar_dataloader(args.dataset,
                                             r=0.0,
                                             noise_mode='none',
                                             batch_size=args.batch_size,
                                             num_workers=4,
                                             root_dir=args.data_path,
                                             log='',
                                             noise_file=os.path.join(args.data_path, 'gt_labels.json'))

        train_val_loader = loader.run('warmup')
        ### extract features
        with torch.no_grad():
            features_all = []
            targets_all = []
            for batch_idx, (inputs, targets) in enumerate(train_val_loader):
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
                print('batch_idx: %04d' % batch_idx, end=', ')
                print('inputs.shape:', inputs.shape, end=', ')
                outputs = resnet(inputs)
                features = outputs.cpu().detach().numpy()
                print('features.shape:', features.shape)
                features_all.append(features)
                targets_all.extend(list(targets.cpu().detach().numpy()))
        features_all = np.concatenate(features_all, axis=0)
        print('features_all.shape: %s' % (features_all.shape,))
        print('len(targets_all):', len(targets_all))
        ### save the extracted features
        resnet_feat['features'] = features_all
        resnet_feat['labels'] = targets_all
        dopickle(resnet_feat, feat_path)

    # for DivideMix, all samples are used for training
    features_train = features_all
    targets_train = targets_all
    print('features_train.shape: %s' % (features_train.shape,))
    print('len(targets_train):', len(targets_train))

    ## K-means++ (use args.seed as the random seed)
    kmeans = KMeans(n_clusters=10, random_state=args.seed).fit(features_train)
    
    ## determine the real label for each cluster
    labels_per_cid = {}
    for i in range(len(targets_train)):
        true_lb = targets_train[i]
        cid = kmeans.labels_[i]
        if cid in labels_per_cid.keys():
            labels_per_cid[cid].append(true_lb)
        else:
            labels_per_cid[cid] = [true_lb]
    cid_to_lb = {}
    for k, v in labels_per_cid.items():
        print('cluster %d has %d elements' % (k, len(v)))
        ### https://stackoverflow.com/questions/20038011/trying-to-find-majority-element-in-a-list
        c = Counter(v)
        # for value, count in c.most_common():
        #     print('(lb %d: %d times)' % (value, count), end=', ')
        # print()
        cid_to_lb[k] = c.most_common()[0][0]
    for k,v in cid_to_lb.items():
        print('cid %d should be assigned with label %d' % (k, v))
    
    ## create noise_label list and save it
    noise_label = []
    noise_count = 0
    for i in range(len(targets_train)):
        lb = int(cid_to_lb[kmeans.labels_[i]])
        noise_label.append(lb)
        if not lb == targets_train[i]:
            noise_count += 1
    json.dump(noise_label, open(noise_file,'w'))
    
    ## compute the noise rate
    print('noise rate: %.4f' % (noise_count / len(targets_train)))

if args.plot_heatmap:
    import matplotlib
    import matplotlib.pyplot as plt
    # plot transition heatmap
    class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_class = len(class_list)
    tm = np.zeros((num_class, num_class))
    for i in range(len(targets_train)):
        tm[targets_train[i]][noise_label[i]] += 1
    row_sums = tm.sum(axis=1, keepdims=True)
    data = tm / row_sums
    print(data)
    
    # create plot
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(data, cmap='Blues')
    
    # set axes
    ax.set_xticks(np.arange(len(class_list)))
    ax.set_yticks(np.arange(len(class_list)))
    # Y-axis will be cut-off on both top and bottom, need the following workaround
    # Ref: https://github.com/matplotlib/matplotlib/issues/14751
    ax.set_ylim(len(class_list)-0.5, -0.5)
    ax.set_xticklabels(class_list, rotation=30)
    ax.set_yticklabels(class_list)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    
    # add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im, cax=cax)
    
    # add text in each cell
    textcolors=('black', 'white')
    threshold = im.norm(data.max())/2.
    kw = dict(horizontalalignment='center', verticalalignment='center')
    valfmt = matplotlib.ticker.StrMethodFormatter('{x:.3f}')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
    
    # save
    heatmap_fname = os.path.basename(noise_file).replace('.json', '_heatmap.png')
    print('produce %s...' % heatmap_fname)
    plt.savefig(heatmap_fname)

